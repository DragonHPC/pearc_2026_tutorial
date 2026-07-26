"""
Fully deterministic counterpart to agent_workflow.py.

Same agentic *pipeline* framework (Dragon Batch DAG + DAGOrchestrator), but
every node is a plain Python function (``PipelineNode(fn=...)``) — no LLM, no
inference service, no agents.  The three nodes mirror the agent workflow::

    check_nans  ──►  choose_new_cfls  ──►  run_experiments
      scans ranks       brackets the          runs the (fake)
      for NaNs          stability limit       simulation

Each node reads/writes the shared data-store DDict and publishes its result
into the orchestrator's DDict exactly like ``run_experiments_node`` does, so
the orchestrator and downstream nodes can consume it.

Usage::
    dragon ultralight_pipeline.py
"""

from functools import partial

import dragon
import multiprocessing as mp

from dragon.ai.agent.config import (
    OrchestratorConfig,
    Pipeline,
    PipelineNode,
    TaskResult,
    TaskStatus,
    DISPATCH_ID_KEY,
    RESULT_KEY,
    STATUS_KEY,
)
from dragon.ai.agent.orchestrator import DAGOrchestrator
from dragon.data.ddict import DDict
from dragon.workflows.batch import Batch

from lightweight_agent_workflow import (
    NUM_RANKS,
    NUM_ITERATIONS,
    CFL_CRITICAL,
    make_scan_all_ranks,
    run_experiments_node,
)


def _read_prev_state(data_store, num_ranks):
    """Return (cfl_values, nan_flags) for the previous round from the DDict.

    ``cfl_values`` is written by the run_experiments node and ``nan_flags`` —
    an aligned list of booleans, one per CFL — by the check_nans node.  A
    missing key is treated as "no prior data" so the proposer falls back to a
    default spread bracketing the stability limit.
    """
    try:
        cfl_values = list(data_store["cfl_values"])
    except KeyError:
        cfl_values = []
    try:
        flags = list(data_store["nan_flags"])
    except KeyError:
        flags = [False] * len(cfl_values)

    # Guard against a length mismatch between the two keys.
    if len(flags) != len(cfl_values):
        flags = (flags + [False] * len(cfl_values))[: len(cfl_values)]
    return cfl_values, flags


def _propose_cfls(cfl_values, flags, num_ranks):
    """Pick num_ranks CFLs bracketing the stability boundary."""
    stable = [c for c, bad in zip(cfl_values, flags) if not bad]
    unstable = [c for c, bad in zip(cfl_values, flags) if bad]
    highest_stable = max(stable) if stable else None
    lowest_nan = min(unstable) if unstable else None
    n = num_ranks

    # Both bounds known: NUM_RANKS points strictly between them.
    if highest_stable is not None and lowest_nan is not None:
        lo, hi = highest_stable, lowest_nan
        if hi <= lo:  # degenerate ordering — nudge just above the stable one
            return [lo + 0.1 * (k + 1) for k in range(n)]
        return [lo + (hi - lo) * (k + 1) / (n + 1) for k in range(n)]

    # Only a stable bound: no NaNs yet, so push ABOVE the highest stable.
    if highest_stable is not None:
        delta = max(0.1, highest_stable * 0.5)
        return [highest_stable + delta * (k + 1) for k in range(n)]

    # Only a NaN bound: everything blew up, so probe BELOW the lowest NaN.
    if lowest_nan is not None:
        delta = lowest_nan / (n + 1)
        return [max(1e-3, lowest_nan - delta * (k + 1)) for k in range(n)]


# ===========================================================================
# Deterministic pipeline nodes (no LLM)
#
# Both are plain functions with the required
# ``(*upstreams: TaskResult) -> TaskResult`` signature once their leading
# arguments (the shared data-store DDict and num_ranks) are bound with
# ``functools.partial`` in run() — exactly like run_experiments_node.
# Each mirrors run_experiments_node's plumbing: attach to the orchestrator's
# DDict via ``upstream.serialized_ddict``, publish DISPATCH/RESULT/STATUS keys
# so downstream nodes (and the orchestrator) can read the result, and return a
# TaskResult marking the node DONE.
# ===========================================================================


def check_nans(data_store, num_ranks, *upstreams: TaskResult) -> TaskResult:
    """Scan all ranks for NaNs and publish the report (no LLM).

    Deterministic replacement for the NaN-detector agent.  It calls the
    ``scan_all_ranks`` tool from agent_workflow (the nan_check tool) to scan
    every rank of every CFL, then records — for each CFL — whether it produced
    NaNs (``nan_flags``, aligned with ``cfl_values``) so the downstream
    ``chose_new_cfls`` node can bracket the stability boundary.

    :param data_store: Shared data-store DDict (bound via partial in run).
    :param num_ranks: Number of MPI ranks per CFL (bound via partial).
    :param upstreams: TaskResult tokens from upstream nodes (the DAG root).
    :returns: A TaskResult marking this node DONE.
    """
    upstream = upstreams[0]
    task_id = upstream.task_id
    serialized_ddict = upstream.serialized_ddict

    ddict = DDict.attach(serialized_ddict)
    try:
        # -- Call the nan_check tool from agent_workflow ----------------------
        scan_all_ranks = make_scan_all_ranks(data_store, num_ranks)
        scan = scan_all_ranks()
        cfls_with_nans = scan["keys_with_nans"]

        # Persist per-CFL NaN flags aligned with cfl_values so the
        # chose_new_cfls node can bracket the stability boundary.
        cfl_values = list(data_store["cfl_values"])
        data_store["nan_flags"] = [c in cfls_with_nans for c in cfl_values]

        print(f"[fn] check_nans -> {scan['report']}", flush=True)

        # -- Publish this node's own result so downstream nodes can run -------
        own_dispatch_id = f"fn-check-nans-{task_id[:8]}"
        ddict[DISPATCH_ID_KEY.format(task_id=task_id, agent_id="check_nans")] = (
            own_dispatch_id
        )
        ddict[
            RESULT_KEY.format(
                task_id=task_id, agent_id="check_nans", dispatch_id=own_dispatch_id
            )
        ] = {"response": scan["report"]}
        ddict[
            STATUS_KEY.format(
                task_id=task_id, agent_id="check_nans", dispatch_id=own_dispatch_id
            )
        ] = TaskStatus.DONE
    finally:
        ddict.detach()

    return TaskResult(
        task_id=task_id,
        agent_id="check_nans",
        status=TaskStatus.DONE,
        serialized_ddict=serialized_ddict,
    )


def chose_new_cfls(data_store, num_ranks, *upstreams: TaskResult) -> TaskResult:
    """Propose CFL numbers by bisection (no LLM).

    Deterministic, model-free replacement for the generator agent.  It reads
    the previous round's CFLs and the per-CFL NaN flags recorded by the
    ``check_nans`` node, brackets the stability limit, and publishes the chosen
    CFLs as a plain comma-separated string — the exact text format
    ``run_experiments_node`` parses via ``_parse_cfls``.

    :param data_store: Shared data-store DDict (bound via partial in run).
    :param num_ranks: Number of MPI ranks per CFL (bound via partial).
    :param upstreams: TaskResult tokens from upstream nodes (check_nans).
    :returns: A TaskResult marking this node DONE.
    """
    upstream = upstreams[0]
    task_id = upstream.task_id
    serialized_ddict = upstream.serialized_ddict

    ddict = DDict.attach(serialized_ddict)
    try:
        # Read the previous round's CFLs + NaN flags (the check_nans node's
        # output), then bracket the stability boundary.
        cfl_values, flags = _read_prev_state(data_store, num_ranks)
        new_cfls = [
            round(float(c), 4) for c in _propose_cfls(cfl_values, flags, num_ranks)
        ]

        # Publish the chosen CFLs as a comma-separated string so the downstream
        # run_experiments node can parse them exactly like the generator
        # agent's free-text answer.
        content = ", ".join(str(c) for c in new_cfls)
        print(f"[fn] chose_new_cfls -> {content}", flush=True)

        own_dispatch_id = f"fn-choose-cfls-{task_id[:8]}"
        ddict[DISPATCH_ID_KEY.format(task_id=task_id, agent_id="choose_new_cfls")] = (
            own_dispatch_id
        )
        ddict[
            RESULT_KEY.format(
                task_id=task_id, agent_id="choose_new_cfls", dispatch_id=own_dispatch_id
            )
        ] = {"response": content}
        ddict[
            STATUS_KEY.format(
                task_id=task_id, agent_id="choose_new_cfls", dispatch_id=own_dispatch_id
            )
        ] = TaskStatus.DONE
    finally:
        ddict.detach()

    return TaskResult(
        task_id=task_id,
        agent_id="choose_new_cfls",
        status=TaskStatus.DONE,
        serialized_ddict=serialized_ddict,
    )


def run(init_cfls, num_ranks, iterations, user_prompt):

    # --- Shared data store -------------------------------------------------
    # A small DDict carries CFL values and per-rank result arrays between the
    # two agents and across search rounds.  Pre-seed each rank so all keys
    # exist before the first experiment runs.
    # See develop/examples/dragon_data/ddict/demo_ddict.py.
    #
    # Three DDicts are in play and EACH must be destroyed, otherwise it leaves
    # a ``ddict_orc_<name>`` metadata file behind in the CWD (a DDict writes
    # that file whenever it is serialized and only removes it on destroy()):
    #   * data_store            -> data_store.destroy()
    #   * the Batch results DDict -> batch.join() (destroys it on last client)
    #   * the DAGOrchestrator DDict -> orchestrator.destroy()
    # They are created before/inside the try and torn down in a single finally
    # so a failure in pipeline/orchestrator construction can never skip a
    # teardown (which previously leaked the Batch results DDict's orc file).
    data_store = None
    batch = None
    orchestrator = None
    try:
        data_store = DDict(1, 1, 2 * 1024 * 1024)
        batch = Batch(managed_lifecycle=True, results_ddict_mem=int(10 * 1024 * 1024))

        partial_run_experiments_node = partial(
            run_experiments_node, batch, data_store.serialize(), num_ranks
        )
        partial_check_nans = partial(check_nans, data_store, num_ranks)
        partial_choose_new_cfls = partial(chose_new_cfls, data_store, num_ranks)

        # Every node is a plain function (fn=...): check_nans scans for NaNs,
        # choose_new_cfls proposes the next CFLs as plain comma-separated text,
        # and run_experiments parses that text and runs the (fake) simulation.
        # No LLM, no agents, no inference service — fully deterministic.
        # This is NOT the recommended way to structure a production pipeline; it is only for demonstration purposes and attempts to align with how the original agent workflow was structured.
        pipeline = Pipeline(
            nodes=[
                PipelineNode(
                    agent_id="check_nans",
                    fn=partial_check_nans,
                    depends_on=[],
                ),
                PipelineNode(
                    agent_id="choose_new_cfls",
                    fn=partial_choose_new_cfls,
                    depends_on=["check_nans"],
                ),
                PipelineNode(
                    agent_id="run_experiments",
                    fn=partial_run_experiments_node,
                    depends_on=["choose_new_cfls"],
                ),
            ]
        )

        # --- Iterative CFL search -----------------------------------------
        # The orchestrator and Batch are created ONCE and reused every round:
        #   * run() may be called repeatedly (a fresh dispatch_id is generated
        #     per call, and global_state is re-seeded each time), and
        #   * a Batch accepts many DAGs; it only becomes unusable after join(),
        #     so we join() a single time after the last round.
        orchestrator = DAGOrchestrator(
            config=OrchestratorConfig(
                agents=[],
                poll_interval=2,
                poll_timeout=1200.0,
            ),
            pipeline=pipeline,
        )

        for iteration in range(iterations):
            if iteration == 0:
                # Bootstrap: run the user-provided CFLs once so the NaN
                # checker has results to scan on the first pipeline pass.
                data_store["cfl_values"] = init_cfls
                run_experiments_node(batch, data_store.serialize(), num_ranks, None)

            print("=" * 60, flush=True)
            print(f"Iteration {iteration + 1}/{iterations}", flush=True)
            print("=" * 60, flush=True)
            print(f"Request: {user_prompt}\n", flush=True)

            # run() returns the terminal node's result — here run_experiments,
            # i.e. a summary of the most recent CFL values that were run.
            result = orchestrator.run(user_input=user_prompt, batch=batch)
            summary = (
                result.get("response", str(result))
                if isinstance(result, dict)
                else str(result)
            )
            print("\n--- Latest run ---", flush=True)
            print(summary, flush=True)

    except Exception as exc:
        import traceback

        print(f"\n[error] CFL search failed: {exc}", flush=True)
        traceback.print_exc()
    finally:
        if orchestrator is not None:
            try:
                orchestrator.destroy()
            except Exception:
                pass
            print("[teardown] Orchestrator destroyed.", flush=True)
        if batch is not None:
            try:
                batch.join()
            except Exception:
                pass
            try:
                batch.destroy(force_timeout=1.0)
            except Exception:
                pass
            print("[teardown] Batch joined (results DDict destroyed).", flush=True)
        if data_store is not None:
            try:
                data_store.destroy()
            except Exception:
                pass
            print("[teardown] Data store (DDict) destroyed.", flush=True)


if __name__ == "__main__":
    mp.set_start_method("dragon")
    init_cfls = [0.3, 2.4, 9.9, 0.8]
    # A single, static command drives every round. The pipeline is
    # nan_agent -> generator_agent -> run_experiments, so each round the
    # NaN checker scans whatever the most recent experiments wrote to the
    # DDict (the bootstrap run below on round 0, or the previous round's
    # run_experiments output), the generator reads that report directly
    # via its DAG dependency and proposes the next CFLs, and
    # run_experiments runs them.  No per-round prompt shuttling is needed.
    user_prompt = (
        "Check the latest CFD result arrays for NaNs and report which "
        "CFL values were stable and which produced NaNs. "
        "Choose the next set of CFL values, increasing CFL for "
        "stable ranks and decreasing it for ranks that produced "
        "NaNs."
    )
    run(init_cfls=init_cfls, num_ranks=2, iterations=1, user_prompt=user_prompt)
