
# ===========================================================================
# Ultralight inference service (model-free)
#
# Drop-in replacement for lightweight_inference_service: identical signature
# and queue protocol, so it can be swapped in without touching any agent
# wiring.  It does NOT load or run a model — it reads the numbers straight
# out of the incoming user message, runs detect_nans directly, and returns
# the agent's final-answer envelope on the response queue.  ``model_name`` is
# accepted only for signature compatibility and is ignored.
#
# Not wired into main() — provided as an example of a zero-latency backend.
# ===========================================================================

def ultralight_nan_service(input_queue, shutdown_event, model_name):
    """Answer NaN queries directly, without an LLM.

    :param input_queue: Shared Dragon Queue the agents publish requests to.
    :param shutdown_event: Dragon Event; set by the parent to stop the loop.
    :param model_name: Ignored — present only for signature compatibility
        with :func:`lightweight_inference_service`.
    """
    import json
    import re

    print("[ultralight] No model loaded — answering by direct NaN check.\n", flush=True)

    while not shutdown_event.is_set():
        try:
            # InferenceRequest is a NamedTuple; index access is robust.
            request = input_queue.get(timeout=1)
        except _pyqueue.Empty:
            continue
        except Exception:
            continue

        messages = request[0]                              # chat messages
        response_queue = request[2]                        # per-request reply

        try:
            # Look only at user-authored text so the agent's own system
            # prompt (which mentions "NaN") can't create false positives.
            user_text = " ".join(
                str(m.get("content", ""))
                for m in messages
                if isinstance(m, dict) and m.get("role") == "user"
            )

            # Prefer the bracketed list, e.g. "[1.0, 2.5, nan, 4.2]"; fall
            # back to the whole message if no list is present.
            match = re.search(r"\[([^\]]*)\]", user_text)
            raw = match.group(1) if match else user_text
            tokens = [t.strip() for t in raw.split(",") if t.strip()]

            # Reuse the exact same checking logic the tool would run.
            result = _check_nans(tokens)

            if result["has_nans"]:
                content = (
                    f"Yes. The data contains {result['nan_count']} NaN "
                    f"value(s) at index/indices {result['nan_indices']} "
                    f"out of {result['total_values']} values."
                )
            else:
                content = (
                    f"No. None of the {result['total_values']} values are NaN."
                )

            # Return the agent's final-answer envelope so the tool-calling
            # loop accepts it as the answer and stops immediately.
            envelope = {"response": {"type": "final_answer", "content": content}}
            response_queue.put({"assistant": json.dumps(envelope)})
        except Exception as exc:  # noqa: BLE001 — report failure to the agent
            response_queue.put(exc)

    print("[ultralight] Shutdown signalled — service stopping.", flush=True)


# ===========================================================================
# Ultralight CFL service (model-free)
#
# Model-free counterpart to ultralight_nan_service, and a drop-in backend for
# the *generator* agent.  Instead of an LLM it reads the previous round's CFL
# values and the per-rank NaN flags — the output the NaN-checking side records
# in the shared DDict (ultralight_nan_service / detect_nans write nan_rank_i)
# — and proposes NUM_RANKS new CFL numbers that bracket the stability limit:
#
#   * between the highest stable CFL and the lowest CFL that produced NaNs;
#   * if no NaN was seen yet, NUM_RANKS values ABOVE the highest stable CFL;
#   * if every rank blew up, NUM_RANKS values BELOW the lowest NaN CFL;
#   * with no prior data at all, a default spread bracketing CFL_CRITICAL.
#
# It then runs the (fake) simulation for the chosen CFLs so the NaN side has
# fresh arrays to check, closing a fully model-free search loop.
#
# Bound to the DDict via a factory (like the make_* tools).  Not wired into
# main() — provided as an example zero-latency backend.
# ===========================================================================

def make_ultralight_cfl_service(data_store):
    """Build a model-free CFL-proposal backend bound to a shared DDict.

    :param data_store: Dragon ``DDict`` shared with the NaN-checking side.
    :returns: An ``ultralight_cfl_service(input_queue, shutdown_event,
        model_name)`` callable usable in place of
        :func:`lightweight_inference_service` for the generator agent.
    """

    def _read_prev_state():
        """Return (cfl_values, nan_flags) for the previous round from the DDict.

        The NaN flag for each rank is taken from the value the NaN-checking
        side recorded (``nan_rank_i``); if that is missing it falls back to
        re-checking the stored array so the service is robust on its own.
        """
        try:
            cfl_values = list(data_store["cfl_values"])
        except KeyError:
            cfl_values = []

        flags = []
        for i in range(len(cfl_values)):
            try:
                flags.append(bool(data_store[f"nan_rank_{i}"]))
            except KeyError:
                try:
                    flags.append(_check_nans(data_store[f"rank_{i}"])["has_nans"])
                except KeyError:
                    flags.append(False)
        return cfl_values, flags

    def _propose_cfls(cfl_values, flags):
        """Pick NUM_RANKS CFLs bracketing the stability boundary."""
        stable = [c for c, bad in zip(cfl_values, flags) if not bad]
        unstable = [c for c, bad in zip(cfl_values, flags) if bad]
        highest_stable = max(stable) if stable else None
        lowest_nan = min(unstable) if unstable else None
        n = NUM_RANKS

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

        # No prior data: default spread bracketing the stability limit.
        lo, hi = 0.5 * CFL_CRITICAL, 1.5 * CFL_CRITICAL
        if n == 1:
            return [CFL_CRITICAL]
        return [lo + (hi - lo) * k / (n - 1) for k in range(n)]

    def ultralight_cfl_service(input_queue, shutdown_event, model_name):
        """Propose CFL numbers without an LLM (see make_ultralight_cfl_service).

        :param input_queue: Shared Dragon Queue the agent publishes requests to.
        :param shutdown_event: Dragon Event; set by the parent to stop the loop.
        :param model_name: Ignored — signature compatibility only.
        """
        import json

        print("[ultralight-cfl] No model loaded — proposing CFLs by bisection.\n",
              flush=True)

        while not shutdown_event.is_set():
            try:
                # InferenceRequest is a NamedTuple; index access is robust.
                request = input_queue.get(timeout=1)
            except _pyqueue.Empty:
                continue
            except Exception:
                continue

            response_queue = request[2]                    # per-request reply

            try:
                # Read the previous round's CFLs + NaN flags (the NaN service's
                # output), then bracket the stability boundary.
                cfl_values, flags = _read_prev_state()
                new_cfls = [round(float(c), 4)
                            for c in _propose_cfls(cfl_values, flags)]

                # Record the chosen CFLs and run the simulation for each rank
                # so the NaN side has fresh arrays to check.
                data_store["cfl_values"] = new_cfls
                for i, cfl in enumerate(new_cfls):
                    data_store[f"rank_{i}"] = _simulate_rank(cfl)

                content = (
                    "Proposed CFL values for the next round: "
                    + ", ".join(f"rank_{i}={c}" for i, c in enumerate(new_cfls))
                )
                envelope = {"response": {"type": "final_answer", "content": content}}
                response_queue.put({"assistant": json.dumps(envelope)})
            except Exception as exc:  # noqa: BLE001 — report failure to the agent
                response_queue.put(exc)

        print("[ultralight-cfl] Shutdown signalled — service stopping.", flush=True)

    return ultralight_cfl_service

## Old junk that probably isn't needed
def make_detect_nans(data_store):
    """Build a ``detect_nans`` tool bound to a shared DDict *data_store*.

    :param data_store: A Dragon ``DDict`` holding the datasets by key.
    :returns: A ``detect_nans(key)`` callable ready for ToolRegistry.register.
    """

    def detect_nans(key: str) -> dict:
        """Determine whether the numeric list stored under a DDict key has NaNs.

        :param key: The DDict key whose value is the list of numbers to
            inspect (e.g. 'sensor_readings').  Non-numeric or missing
            entries are also reported as invalid (NaN-like).
        :returns: Dict with keys 'has_nans' (bool), 'nan_count' (int),
            'nan_indices' (list of positions), and 'total_values' (int).
        """
        data = data_store[key]          # value fetched from the shared DDict
        result = _check_nans(data)
        # Persist the per-rank flag so the generator agent can react to it
        # in the next round of the CFL search.
        data_store[f"nan_{key}"] = result["has_nans"]
        print(f"[tool] detect_nans(key={key!r}) -> {result}", flush=True)
        return result

    return detect_nans



def run_cfl_experiments(*upstreams: TaskResult) -> TaskResult:
    """Run one CFD experiment per MPI rank at the given CFL numbers.

    Stores the chosen CFL values and writes each rank's result array to
    the shared store under 'rank_0'..'rank_{N-1}'.  A rank whose CFL is
    above the solver's stability limit produces NaNs.

    :param cfl_values: CFL numbers to test, one per rank.  At most
        NUM_RANKS values are used; if fewer are given the last value is
        repeated to fill NUM_RANKS ranks.
    :returns: Dict summarising the experiments that were launched.
    """
    upstream = upstreams[0]
    task_id = upstream.task_id
    serialized_ddict = upstream.serialized_ddict

    ddict = DDict.attach(serialized_ddict)

    data_ddict_ser = ddict["data_ddict_ser"]
    cfl_values = ddict["cfl_values"]
    num_ranks = ddict["num_ranks"]

    vals = [float(c) for c in cfl_values]

    ddict["cfl_values"] = vals
    for i, cfl in enumerate(vals):
        # this function simulates the CFD result for the given CFL number on this num_ranks. Each rank writes the values for it's part of the grid into the DDict at 'cfl_{val}_rank_{i}'.
        _simulate_rank(ddict, cfl, num_ranks)
    # maybe some batch call for blocking to wait on these jobs?

    summary = {
        "num_experiments": len(vals),
        "cfl_values": vals,
        "rank_keys": [f"cfl_{cfl}_rank_{i}" for cfl in vals for i in range(num_ranks)],
    }
    print(f"[tool] run_cfl_experiments -> {summary}", flush=True)
    ddict["summary"] = summary

    return TaskResult(
        task_id=task_id,
        agent_id="save_report",
        status=TaskStatus.DONE,
        serialized_ddict=serialized_ddict,
    )