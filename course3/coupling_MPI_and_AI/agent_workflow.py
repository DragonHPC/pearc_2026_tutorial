"""Two-agent CFL search: generator + NaN detector on Dragon.

This example combines two ideas from the tutorial:

* The **Dragon AI agent framework** used in
  ``develop/examples/dragon_ai/ai_agent/01_single_agent.py`` and
  ``02_multi_agent_dag.py`` (``create_sub_agent``, ``ToolRegistry``,
  ``DAGOrchestrator``, ``Pipeline``).
* A **lightweight HuggingFace-transformers inference service** — the same
  idea as ``inference_service_light`` in
  ``course3/coupling_MPI_and_AI/agents.py`` — used *instead of*
  ``dragon.ai.inference`` (the vLLM-based pipeline).

Why this works
--------------
An agent talks to its LLM backend through a single Dragon Queue.  Internally
the agent wraps that queue in a ``DragonQueueLLMProxy`` which, for every
chat call, puts an ``InferenceRequest`` tuple on the queue::

    InferenceRequest(
        messages,            # [0] OpenAI-format chat messages
        formatted_messages,  # [1]
        response_queue,      # [2] per-request reply queue
        timestamp,           # [3]
        tools,               # [4] tool JSON schemas (or None)
        sampling_override,   # [5]
        continue_final_message,  # [6]
        stream,              # [7]
    )

...and then blocks on ``response_queue.get()``.  The backend only has to
return a ``{"assistant": <text>}`` dict (or a plain string) on that queue.
That is exactly what ``lightweight_inference_service`` below does with a
local ``transformers`` model — no vLLM, no GPU required.

Architecture::

    Main Process
      └── lightweight_inference_service (Dragon Process)
            └── AutoModelForCausalLM.generate()   ← plain transformers

    Iterative CFL search (looped)::
        generator_agent  ──►  nan_agent
          │ run_cfl_experiments   │ detect_nans (per rank)
          ▼                       ▼
        picks CFLs + simulates    flags which ranks blew up

    Each round the generator uses the previous round's NaN results to push
    toward the largest CFL that does not produce NaNs.

Usage::

    dragon nan_detector_agent.py
"""

import asyncio
from functools import partial
import math
import os
import queue as _pyqueue
from pathlib import Path
from ai_cfd_workflow import queue_job

import dragon
import multiprocessing as mp

from dragon.ai.agent.core import create_sub_agent
from dragon.ai.agent.config import (
    AgentConfig,
    OrchestratorConfig,
    Pipeline,
    PipelineNode,
    TaskResult,
    TaskStatus,
    DISPATCH_ID_KEY,
    RESULT_KEY,
    STATUS_KEY,
)
from dragon.ai.agent.tools import ToolRegistry
from dragon.ai.agent.orchestrator import DAGOrchestrator
from dragon.data.ddict import DDict
from dragon.native.event import Event
from dragon.native.process import Process
from dragon.native.queue import Queue
from dragon.workflows.batch import Batch

NUM_RANKS = 4
NUM_ITERATIONS = 2          # number of CFL search rounds
CFL_CRITICAL = 1.0          # hidden stability limit: CFL > this -> NaNs
# ===========================================================================
# Model location
#
# Defaults to the repository's ``model/`` directory (SmolLM3), matching the
# convention in agents.py.  Override with DRAGON_LOCAL_MODEL_DIR.
# ===========================================================================

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_DIR = _REPO_ROOT / "model"
LOCAL_MODEL_DIR = os.environ.get(
    "DRAGON_LOCAL_MODEL_DIR",
    str(_DEFAULT_MODEL_DIR if _DEFAULT_MODEL_DIR.exists() else Path("/tmp/model")),
)

# Max tokens the model may generate per LLM call.  NOTE: this is *not* the
# length of the final yes/no answer — the agent's tool-calling loop makes the
# model emit JSON envelopes (a tool_request that echoes the full input list,
# then a final_answer).  Setting this too low (e.g. 4) truncates that JSON and
# breaks the reasoning loop.  Lower it only as far as your prompts allow.
MAX_NEW_TOKENS = int(os.environ.get("DRAGON_MAX_NEW_TOKENS", "1024"))


# ===========================================================================
# Tool implementation
#
# Rather than passing the whole data list through the prompt text, the agent
# passes only a *key*.  The list itself lives in a shared Dragon Distributed
# Dictionary (DDict); the tool looks it up by key.  This keeps the LLM context
# small and lets large datasets be shared by reference (see demo_ddict.py).
#
# ``make_detect_nans(data_store)`` binds the DDict into a ``detect_nans(key)``
# closure.  register() then derives the tool name/description/param schema from
# the closure's __name__, __doc__, and type annotations.  Dragon cloudpickles
# the closure (and the captured DDict handle) to the agent process, where the
# DDict auto-attaches, so ``data_store[key]`` works there transparently.
# ===========================================================================

def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` JSON object substring, or None.

    Greedy decoding sometimes appends stray tokens after a complete envelope
    (especially when we prime a ``tool_request`` — the model closes the JSON
    and then keeps going).  ``json.loads`` rejects that trailing text, so we
    scan for the first brace-balanced object (respecting string literals and
    escapes) and hand just that slice to the parser.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _coerce_to_envelope(text: str) -> str:
    """Normalize raw model output into a schema-valid response envelope.

    This backend does NOT enforce guided decoding, so a small model may
    break the required JSON contract — most commonly by emitting free-form
    prose (or a ``<think>`` trace / markdown fence) on the turn where it
    means to give its final answer.  The agent's parser then sees no valid
    ``{"response": {...}}`` object, manufactures an *empty* ``final_answer``,
    and the agent returns no text.

    To make the tool-calling loop robust against that, we:

    1. Strip any ``<think>...</think>`` block and ```` ``` ```` code fences.
    2. If what remains parses as JSON containing a ``"response"`` key, pass
       it through unchanged (a well-formed tool_request or final_answer).
    3. Otherwise treat the text as the model's final answer and wrap it in a
       ``final_answer`` envelope so the loop terminates with real content.

    :param text: Raw decoded model output.
    :returns: A JSON string that satisfies the agent's ResponseModel schema.
    """
    import json
    import re

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip a leading ```json / ``` fence and trailing ``` if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    # Case 2: already a valid response envelope — leave it untouched.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "response" in parsed:
            return cleaned
    except (json.JSONDecodeError, TypeError):
        pass

    # Case 2b: a valid envelope followed by trailing tokens — common when we
    # prime a tool_request and greedy decoding keeps generating past the
    # closing braces.  Extract the first balanced object and use it if it
    # validates as a response envelope.
    candidate = _extract_first_json_object(cleaned)
    if candidate is not None:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "response" in parsed:
                return candidate
        except (json.JSONDecodeError, TypeError):
            pass

    # Case 3: prose (or malformed JSON) — wrap as a final answer so the
    # agent's loop accepts it and stops instead of returning empty text.
    envelope = {
        "response": {"type": "final_answer", "content": cleaned or text.strip()}
    }
    return json.dumps(envelope)


def _check_nans(values: list) -> dict:
    """Core NaN check shared by the tool and the ultralight service."""
    nan_indices = []
    for i, value in enumerate(values):
        try:
            if math.isnan(float(value)):
                nan_indices.append(i)
        except (TypeError, ValueError):
            # None, strings, etc. can't be a valid number → treat as NaN-like.
            nan_indices.append(i)

    max_value = max(values)
    min_value = min(values)
    considered_nans = max_value > 10**6 or min_value < -10**6
    return {
        "has_nans": len(nan_indices) > 0 or considered_nans,
        "max_value": max_value,
        "min_value": min_value,
    }


def make_scan_all_ranks(data_store, num_ranks: int):
    """Build a zero-argument ``scan_all_ranks`` tool bound to a DDict.

    ``ToolRegistry.register`` derives the tool's name, description, and JSON
    parameter schema from the callable's ``__name__``, ``__doc__``, and
    ``inspect.signature`` (see ``FunctionTool``).  A ``functools.partial``
    exposes none of those, so registering one raises
    ``AttributeError: 'functools.partial' object has no attribute '__name__'``.
    Returning a *named* closure fixes that: ``data_store`` and ``num_ranks``
    are captured (and cloudpickled to the agent process, where the DDict
    auto-attaches), leaving a tool the LLM invokes with no arguments.
    """

    def scan_all_ranks() -> dict:
        """Scan every rank array for NaNs and return a complete report.

        Checks keys in the shared data store in a single call, persists each rank's NaN flag, and reports which ranks blew up (NaNs) and which were stable.  Use this tool exactly once, then report its 'report' field as your final answer.

        :returns: Dict with 'keys_with_nans' (CFLs that produced NaNs),
            'keys_without_nans' (stable CFLs), and 'report' (a ready-to-use
            human-readable summary string covering every rank).
        """

        cfl_values = list(data_store["cfl_values"])
        cfls_with_nans = []
        cfls_without_nans = []
        for cfl in cfl_values:
            keyp1 = f"cfl_{cfl}"
            for i in range(num_ranks):
                # Must match the key the MPI solver writes in mpi4py_example.py:
                # f"cfl_{cfl}_rank{rank}" (no underscore before the rank index).
                key = keyp1 + f"_rank{i}"
                result = _check_nans(data_store[key])
                print(f"[tool] Result for key: {key}: {result}", flush=True)
                if result["has_nans"]:
                    cfls_with_nans.append(cfl)
                    break
            else:
                cfls_without_nans.append(cfl)

        report = (
            f"Stable CFLs (no NaNs): {cfls_without_nans or 'none'}. "
            f"Unstable CFLs (produced NaNs): {cfls_with_nans or 'none'}."
        )
        out = {
            "keys_with_nans": cfls_with_nans,
            "keys_without_nans": cfls_without_nans,
            "report": report,
        }
        print(f"[tool] scan_all_ranks() -> {out}", flush=True)
        return out

    return scan_all_ranks

# ===========================================================================
# Experiment execution as a plain-function DAG node (no LLM)
#
# Instead of asking the generator LLM to *call* run_cfl_experiments (an extra,
# error-prone tool turn), the generator only proposes CFL numbers as free
# text.  This function node sits BETWEEN the generator and the NaN agent in
# the DAG: it reads the generator's proposed CFLs from the shared DDict, runs
# the (fake) simulation deterministically, and writes the per-rank arrays.
#
# A PipelineNode(fn=...) runs as a plain Python function in Dragon Batch:
#   * it receives the upstream nodes' TaskResult tokens,
#   * attaches to the shared DDict via upstream.serialized_ddict,
#   * and writes its own DISPATCH_ID/RESULT/STATUS keys so downstream nodes
#     (and the orchestrator) can see it completed.
# See develop/examples/dragon_ai/ai_agent/02_multi_agent_dag.py (save_report).
# ===========================================================================

def _parse_cfls(text: str, num_ranks: int) -> list:
    """Extract up to *num_ranks* CFL numbers from the generator's free text.

    The generator now returns plain text (e.g. "0.4, 0.8, 1.2, 1.6"), so this
    parser must be forgiving.  It strips ``rank_<n>`` tokens first so rank
    indices are not mistaken for CFL values, prefers decimal numbers over bare
    integers, keeps only plausible CFL magnitudes, and falls back to a
    deterministic spread bracketing the stability limit if nothing usable is
    found — so a weak or malformed model reply can never crash the workflow.

    :param text: The generator agent's final answer text.
    :param num_ranks: Number of CFL values to return (one per rank).
    :returns: A list of *num_ranks* floats.
    """
    import re

    # Drop 'rank_0', 'rank 1', etc. so their indices aren't parsed as CFLs.
    cleaned = re.sub(r"rank[_\s]*\d+", " ", text, flags=re.IGNORECASE)
    tokens = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", cleaned)

    decimals = [float(t) for t in tokens if "." in t]
    pool = decimals if decimals else [float(t) for t in tokens]
    vals = [v for v in pool if 0 < v <= 10][:num_ranks]

    if not vals:
        # Deterministic fallback: spread values bracketing the stability limit.
        lo, hi = 0.5 * CFL_CRITICAL, 1.5 * CFL_CRITICAL
        if num_ranks == 1:
            return [round(CFL_CRITICAL, 4)]
        return [round(lo + (hi - lo) * k / (num_ranks - 1), 4)
                for k in range(num_ranks)]

    while len(vals) < num_ranks:      # pad short lists by repeating the last
        vals.append(vals[-1])
    return [round(v, 4) for v in vals]



def run_experiments_node(batch, data_ddict_ser, num_ranks, *upstreams: TaskResult) -> TaskResult:
    """Read the generator's proposed CFLs and run one experiment per rank.

    :param upstreams: TaskResult tokens from upstream nodes (the generator).
    :returns: A TaskResult marking this node DONE.
    """
    # ``*upstreams`` is always a tuple, so it is never None.  The first,
    # direct call passes an explicit None to signal the initial (user-seeded)
    # run, so detect the pipeline case by inspecting the first element.
    pipeline_run = bool(upstreams) and upstreams[0] is not None

    data_ddict = DDict.attach(data_ddict_ser)
    ddict = None
    task_id = None
    serialized_ddict = None
    try:
        # we're running in the pipeline so got cfls from previous agent's output
        if pipeline_run:
            upstream = upstreams[0]
            task_id = upstream.task_id
            serialized_ddict = upstream.serialized_ddict

            ddict = DDict.attach(serialized_ddict)
            # -- Read the generator agent's proposed CFL text from the DDict --
            gen_dispatch_id = ddict[
                DISPATCH_ID_KEY.format(task_id=task_id, agent_id="generator_agent")
            ]
            gen_result = ddict[
                RESULT_KEY.format(
                    task_id=task_id,
                    agent_id="generator_agent",
                    dispatch_id=gen_dispatch_id,
                )
            ]
            gen_text = (
                gen_result.get("response", str(gen_result))
                if isinstance(gen_result, dict) else str(gen_result)
            )
            cfls = _parse_cfls(gen_text, num_ranks)
            data_ddict["cfl_values"] = cfls
        # it's the first run so we're running with user provided cfls
        else:
            cfls = data_ddict["cfl_values"]

        print(f"[fn] run_experiments_node -> cfl_values={cfls}", flush=True)
        jobs = []  # collect all jobs for potential batch processing
        for cfl in cfls:
            # this function simulates the CFD result for the given CFL number on this num_ranks. Each rank writes the values for it's part of the grid into the DDict at 'cfl_{val}_rank_{i}'.
            uid, job = queue_job(batch, num_ranks, cfl, data_ddict_ser)
            jobs.append(job)

        for job in jobs:
            try:
                ecodes = job.get()
                print(f"Got ecodes: {ecodes}", flush=True)
            except Exception as e:
                print(f"Got exception from MPI job: {e}", flush=True)

        if pipeline_run:
            for key, val in ddict.items():
                print(f"[fn] run_experiments_node -> {key}={val}", flush=True)

            # -- Publish this node's own result so downstream nodes can run ---
            own_dispatch_id = f"fn-run-experiments-{task_id[:8]}"
            ddict[DISPATCH_ID_KEY.format(
                task_id=task_id, agent_id="run_experiments")] = own_dispatch_id
            ddict[RESULT_KEY.format(
                task_id=task_id, agent_id="run_experiments",
                dispatch_id=own_dispatch_id)] = {
                "response": f"Ran experiments at CFL values {cfls}."
            }
            ddict[STATUS_KEY.format(
                task_id=task_id, agent_id="run_experiments",
                dispatch_id=own_dispatch_id)] = TaskStatus.DONE
    finally:
        if ddict is not None:
            ddict.detach()
        data_ddict.detach()

    return TaskResult(
        task_id=task_id,
        agent_id="run_experiments",
        status=TaskStatus.DONE,
        serialized_ddict=serialized_ddict,
    )

# ===========================================================================
# Lightweight inference service
#
# Drop-in replacement for dragon.ai.inference.Inference.  Runs as its own
# Dragon Process and speaks the same queue protocol the agent expects:
#   1. read an InferenceRequest tuple from the shared inference queue
#   2. apply the model's chat template (with tool schemas, if any)
#   3. generate with a local transformers model
#   4. put {"assistant": <text>} on the request's response queue
# ===========================================================================

def lightweight_inference_service(input_queue, shutdown_event, model_name):
    """Serve agent LLM requests using a local HuggingFace model.

    :param input_queue: Shared Dragon Queue the agents publish requests to.
    :param shutdown_event: Dragon Event; set by the parent to stop the loop.
    :param model_name: Path or name of the transformers checkpoint to load.
    """
    import dragon.ai.torch  # noqa: F401  — Dragon-aware torch shims
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    print(f"[inference] Loading model from {model_name} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,   # avoid holding two weight copies during load
    ).to(device)
    model.eval()
    print("[inference] Model ready — serving requests.\n", flush=True)

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
        tools = request[4] if len(request) > 4 else None   # tool schemas

        try:
            # Build the prompt with the model's chat template.  Pass the tool
            # schemas so the model knows what it may call, and disable the
            # SmolLM3 "thinking" trace so we get direct JSON tool decisions.
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:
                # Fall back for templates without tools/enable_thinking kwargs.
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )

            # --- Force the first tool-using turn to be a real tool call ------
            # This backend ignores the guided-decoding json_schema hint, so a
            # small model often skips straight to a fabricated final_answer
            # (e.g. content="<tool_result_from_scan_all_ranks>") without ever
            # invoking the tool.  When tools are available AND no tool result
            # has come back yet, prefill the assistant turn with the opening of
            # a tool_request envelope; greedy decoding then continues from the
            # primer and emits an actual tool call.  The primer stops at the
            # tool "name" so the model still chooses which tool to invoke.
            #
            # Real tool results are appended by the agent loop as
            # {"role": "tool", ...} messages, so their presence is the signal
            # that the model has what it needs and may now answer — in which
            # case we do NOT prime and let it produce a final_answer.
            primer = ""
            tool_result_seen = any(
                isinstance(m, dict) and m.get("role") == "tool" for m in messages
            )
            if tools and not tool_result_seen:
                primer = '{"response": {"type": "tool_request", "tool_calls": [{"name": "'
                prompt_text += primer

            model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,           # greedy → stable JSON output
                )
            new_ids = generated[0][len(model_inputs.input_ids[0]):]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            # Re-attach the primer so the returned text is the complete
            # envelope (the model only generated the continuation).
            if primer:
                text = primer + text

            # --- DEBUG: show exactly what the model emitted --------------------
            # The agent's tool-calling loop requires STRICT schema-valid JSON
            # (ResponseModel.model_validate_json).  This backend does NOT honor
            # the json_schema/guided-decoding hint (request[5]), so the model
            # is free to emit <think> traces, markdown fences, or prose — any of
            # which make the parser fail and the agent produce no final answer.
            # Print the raw output so we can see whether it is valid JSON.
            print(
                f"[inference][debug] schema_hint={'yes' if len(request) > 5 and request[5] else 'no'} "
                f"tools={'yes' if tools else 'no'} raw_output={text!r}",
                flush=True,
            )

            # Coerce non-schema output (prose / think trace / fenced JSON) into
            # a valid response envelope so a format-breaking "final" turn still
            # terminates the agent loop with real text instead of empty output.
            #
            # Only do this for tool-using agents (the structured-output loop).
            # A tool-less agent (e.g. the CFL generator) does a plain chat and
            # expects its raw text back — wrapping it would double-encode it.
            if tools:
                payload = _coerce_to_envelope(text)
            else:
                payload = text

            print(f"[inference][debug] payload={payload!r}", flush=True)
            # The agent's DragonQueueLLMProxy accepts a dict with "assistant".
            response_queue.put({"assistant": payload})
        except Exception as exc:  # noqa: BLE001 — report failure to the agent
            # Returning the exception lets the proxy re-raise it agent-side.
            response_queue.put(exc)

    print("[inference] Shutdown signalled — inference service stopping.", flush=True)

# ===========================================================================
# Main
# ===========================================================================

def main(init_cfls, num_ranks):
    input_queue = Queue()
    inference_shutdown = Event()

    # --- Shared data store -------------------------------------------------
    # A small DDict carries CFL values and per-rank result arrays between the
    # two agents and across search rounds.  Pre-seed each rank so all keys
    # exist before the first experiment runs.
    # See develop/examples/dragon_data/ddict/demo_ddict.py.
    data_store = DDict(1, 1, 2 * 1024 * 1024)
    batch = Batch(results_ddict_mem=int(10 * 1024 * 1024))

    partial_run_experiments_node = partial(run_experiments_node, batch, data_store.serialize(), num_ranks)
    scan_all_ranks_tool = make_scan_all_ranks(data_store, num_ranks)

    # Generator agent proposes CFL numbers as plain text (NO tools) — the
    # run_experiments function node in the pipeline parses that list and runs
    # the simulation, so the LLM never has to make a tool call.
    generator_registry = ToolRegistry()
    nan_registry = ToolRegistry()

    # Single-call tool: checks every rank in one shot, refreshes all
    # nan_rank_i flags, and returns a ready-made report.  This keeps the
    # agent's job to "call once, relay the report", which small models
    # perform far more reliably than an N-step detect_nans loop.
    nan_registry.register(scan_all_ranks_tool)

    print("[startup] Launching lightweight inference service...", flush=True)
    #CPW: maybe something for them to plug in
    inference_proc = Process(
        target=lightweight_inference_service,
        args=(input_queue, inference_shutdown, LOCAL_MODEL_DIR),
    )
    inference_proc.start()

    procs, agent_specs = [], []
    try:
        pipeline = Pipeline(nodes=[
            # Plain-function node (no LLM): parses the generator's proposed CFLs and
            # runs the experiments deterministically, so the generator never has to
            # make a tool call.
            PipelineNode(
                agent_id="nan_agent",
                task_description=(
                    "You are a data-quality assistant.  A CFD experiment has just "
                    f"written result arrays for N MPI ranks into the Distributed Dictionary (DDict)."
                    "WORKFLOW (do this exactly):\n"
                    "  1. Call the scan_all_ranks tool ONCE.  It takes no arguments "
                    "and checks every rank in a single call.\n"
                    "  2. Take the 'report' string from the tool result and return it "
                    "verbatim as your final answer.  Do NOT call any tool again and "
                    "do NOT invent numbers — use only the tool's output."
                ),
                depends_on=[],
            ),
            PipelineNode(
                agent_id="generator_agent",
                task_description=(
                    "You are a CFL (Courant number) search planner for a CFD solver "
                    f"running on N MPI ranks.  Your goal is to find the "
                    "LARGEST CFL number that does NOT make the solver blow up "
                    "(produce NaNs). The message tells you the previous round's results (the CFL tested and whether that it produced NaNs)\n\n"
                    "Decision rules:\n"
                    "INCREASE the CFL for ranks that were stable and "
                    "DECREASE it for ranks that produced NaNs, narrowing toward the "
                    "largest stable CFL.\n\n"
                    "OUTPUT FORMAT:\n"
                    f"  Respond with a list of new CFL numbers (one per rank) as a "
                    "plain comma-separated list, e.g. '0.4, 0.8, 1.2, 1.6'.  Do not "
                    "call any tools and do not add any other text."
                ),
                depends_on=["nan_agent"],
            ),
            PipelineNode(
                agent_id="run_experiments",
                fn=partial_run_experiments_node,
                depends_on=["generator_agent"],
            ),
        ])
        # --- Create the two agents: generator + NaN checker ---
        agent_specs = [
            {
                "config": AgentConfig(
                    agent_id="generator_agent",
                    name="CFL Generator",
                    role=(
                        "You choose CFL numbers for a CFD solver, seeking the "
                        "largest CFL that does not blow up (produce NaNs).  You "
                        "reply with only a comma-separated list of numbers."
                    ),
                    inference_queue=input_queue,
                    max_concurrent_requests=1,
                    # No tools: the generator makes a single plain LLM call and
                    # returns its CFL list as text (no tool-calling loop).
                    max_tool_call_iterations=6,
                ),
                "tool_registry": generator_registry,
                "shutdown_event": Event(),
                "reply_queue": Queue(),
            },
            {
                "config": AgentConfig(
                    agent_id="nan_agent",
                    name="NaN Detector",
                    role=(
                        "You are a data-quality assistant.  You call the "
                        "scan_all_ranks tool exactly once and report its "
                        "'report' string verbatim as your answer."
                    ),
                    inference_queue=input_queue,
                    max_concurrent_requests=1,
                    # One scan_all_ranks call + one final answer is all that is
                    # needed; leave a little headroom for a retry.
                    max_tool_call_iterations=6,
                ),
                "tool_registry": nan_registry,
                "shutdown_event": Event(),
                "reply_queue": Queue(),
            },
        ]

        # Launch each agent as a Dragon Process.
        for spec in agent_specs:
            p = Process(target=create_sub_agent, kwargs=spec)
            p.start()
            procs.append(p)

        # Each agent publishes its own input queue once it is ready.
        for spec in agent_specs:
            spec["config"].input_queue = spec["reply_queue"].get()
            print(f"[startup] Agent '{spec['config'].agent_id}' ready.", flush=True)
        print(flush=True)

        # --- Iterative CFL search -----------------------------------------
        # The orchestrator and Batch are created ONCE and reused every round:
        #   * run() may be called repeatedly (a fresh dispatch_id is generated
        #     per call, and global_state is re-seeded each time), and
        #   * a Batch accepts many DAGs; it only becomes unusable after join(),
        #     so we join() a single time after the last round.
        orchestrator = DAGOrchestrator(
            config=OrchestratorConfig(
                agents=[s["config"] for s in agent_specs],
                poll_interval=2,
                poll_timeout=1200.0,
            ),
            pipeline=pipeline,
        )

        try:
            # A single, static command drives every round.  The pipeline is
            # nan_agent -> generator_agent -> run_experiments, so each round the
            # NaN checker scans whatever the most recent experiments wrote to the
            # DDict (the bootstrap run below on round 0, or the previous round's
            # run_experiments output), the generator reads that report directly
            # via its DAG dependency and proposes the next CFLs, and
            # run_experiments runs them.  No per-round prompt shuttling is needed.
            user_input = (
                "Check the latest CFD result arrays for NaNs and report which "
                "CFL values were stable and which produced NaNs. "
                "Choose the next set of CFL values, increasing CFL for "
                "stable ranks and decreasing it for ranks that produced "
                "NaNs."
            )

            for iteration in range(NUM_ITERATIONS):
                if iteration == 0:
                    # Bootstrap: run the user-provided CFLs once so the NaN
                    # checker has results to scan on the first pipeline pass.
                    data_store["cfl_values"] = init_cfls
                    run_experiments_node(batch, data_store.serialize(),  num_ranks, None)

                print("=" * 60, flush=True)
                print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}", flush=True)
                print("=" * 60, flush=True)
                print(f"Request: {user_input}\n", flush=True)

                # run() returns the terminal node's result — here run_experiments,
                # i.e. a summary of the most recent CFL values that were run.
                result = orchestrator.run(user_input=user_input, batch=batch)
                summary = (
                    result.get("response", str(result))
                    if isinstance(result, dict) else str(result)
                )
                print("\n--- Latest run ---", flush=True)
                print(summary, flush=True)

        except Exception as exc:
            import traceback
            print(f"\n[error] CFL search failed: {exc}", flush=True)
            traceback.print_exc()
        finally:
            orchestrator.destroy()
            batch.join()

    except Exception as exc:
        import traceback
        print(f"\n[error] Fatal: {exc}", flush=True)
        traceback.print_exc()
    finally:
        for spec in agent_specs:
            try:
                spec["shutdown_event"].set()
            except Exception:
                pass
        for p in procs:
            try:
                p.join()
            except Exception:
                pass
        print("\n[teardown] Agent stopped.", flush=True)

        inference_shutdown.set()
        try:
            inference_proc.join()
        except Exception:
            pass
        print("[teardown] Inference service stopped.", flush=True)

        try:
            data_store.destroy()
        except Exception:
            pass
        print("[teardown] Data store (DDict) destroyed.", flush=True)


if __name__ == "__main__":
    mp.set_start_method("dragon")
    init_cfls = [0.3, 2.4, 9.9, 0.8]
    main(init_cfls=init_cfls, num_ranks=NUM_RANKS)
