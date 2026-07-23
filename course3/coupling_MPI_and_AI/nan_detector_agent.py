"""Dragon agent + lightweight (transformers) inference — NaN detector.

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

    nan_agent (Dragon Process)
      └── receives task via Dragon Queue
      └── calls detect_nans tool
      └── returns result

Usage::

    dragon nan_detector_agent.py
"""

import asyncio
import math
import os
import queue as _pyqueue
from pathlib import Path

import dragon
import multiprocessing as mp

from dragon.ai.agent.core import create_sub_agent
from dragon.ai.agent.config import (
    AgentConfig,
    OrchestratorConfig,
    Pipeline,
    PipelineNode,
)
from dragon.ai.agent.tools import ToolRegistry
from dragon.ai.agent.orchestrator import DAGOrchestrator
from dragon.data.ddict import DDict
from dragon.native.event import Event
from dragon.native.process import Process
from dragon.native.queue import Queue
from dragon.workflows.batch import Batch


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

    return {
        "has_nans": len(nan_indices) > 0,
        "nan_count": len(nan_indices),
        "nan_indices": nan_indices,
        "total_values": len(values),
    }


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
        print(f"[tool] detect_nans(key={key!r}) -> {result}", flush=True)
        return result

    return detect_nans


def make_scan_all_ranks(data_store, num_ranks: int):
    """Build a ``scan_all_ranks`` tool that checks every rank in one call.

    Small models are unreliable at multi-step tool loops (call detect_nans,
    observe, call again, ..., then format a final answer).  They tend to stop
    after the first call and hallucinate the rest — exactly what SmolLM3 does
    here.  Collapsing the work into a **single** no-argument tool call turns
    the agent's job into "call once, then report", which small models handle
    far more reliably.  The tool also returns a ready-made ``report`` string,
    so even a weak final turn just relays correct, tool-derived text.

    :param data_store: A Dragon ``DDict`` holding per-rank arrays by key.
    :param num_ranks: Number of ranks (keys are 'rank_0' .. 'rank_{N-1}').
    :returns: A ``scan_all_ranks()`` callable ready for ToolRegistry.register.
    """

    def scan_all_ranks() -> dict:
        """Scan every rank array for NaNs and return a complete report.

        Checks keys 'rank_0' through 'rank_{N-1}' in the shared data store
        in a single call.  Use this tool exactly once, then report its
        'report' field as your final answer.

        :returns: Dict with 'keys_with_nans' (list of rank keys that contain
            NaNs), 'per_key' (per-rank NaN details), and 'report' (a
            ready-to-use human-readable summary string).
        """
        per_key = {}
        keys_with_nans = []
        for i in range(num_ranks):
            key = f"rank_{i}"
            result = _check_nans(data_store[key])
            per_key[key] = result
            if result["has_nans"]:
                keys_with_nans.append(key)

        if keys_with_nans:
            parts = [
                f"{k} has {per_key[k]['nan_count']} NaN(s) at "
                f"indices {per_key[k]['nan_indices']}"
                for k in keys_with_nans
            ]
            report = "; ".join(parts) + "."
        else:
            report = "No ranks contain NaN values."

        out = {
            "keys_with_nans": keys_with_nans,
            "per_key": per_key,
            "report": report,
        }
        print(f"[tool] scan_all_ranks() -> {out}", flush=True)
        return out

    return scan_all_ranks


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

            model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,           # greedy → stable JSON output
                )
            new_ids = generated[0][len(model_inputs.input_ids[0]):]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

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
            envelope = _coerce_to_envelope(text)

            # The agent's DragonQueueLLMProxy accepts a dict with "assistant".
            response_queue.put({"assistant": envelope})
        except Exception as exc:  # noqa: BLE001 — report failure to the agent
            # Returning the exception lets the proxy re-raise it agent-side.
            response_queue.put(exc)

    print("[inference] Shutdown signalled — inference service stopping.", flush=True)


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
# Pipeline — single node, single agent
#
# The tool registry is built inside main() because the tool must be bound to
# a DDict that only exists once the Dragon runtime is up.
# ===========================================================================

pipeline = Pipeline(nodes=[
    PipelineNode(
        agent_id="nan_agent",
        task_description=(
            "You are a data-quality assistant.  Per-rank result arrays are "
            "stored in the shared data store.\n\n"
            "WORKFLOW (do this exactly):\n"
            "  1. Call the scan_all_ranks tool ONCE.  It takes no arguments "
            "and checks every rank in a single call.\n"
            "  2. Take the 'report' string from the tool result and return it "
            "verbatim as your final answer.  Do NOT call any tool again and "
            "do NOT invent numbers — use only the tool's output."
        ),
        depends_on=[],
    ),
])


# ===========================================================================
# Main
# ===========================================================================

async def main():
    input_queue = Queue()
    inference_shutdown = Event()

    # --- Shared data store -------------------------------------------------
    # Create a small DDict (1 manager, 1 node, 2 MB) and stage the datasets
    # under keys.  The payload here is tiny, so there is no need to reserve a
    # large managed pool.  The agent passes a key to the detect_nans tool
    # instead of the raw list.
    # See develop/examples/dragon_data/ddict/demo_ddict.py.
    data_store = DDict(1, 1, 2 * 1024 * 1024)

    # Simulate result arrays gathered from four MPI ranks (rank_0..rank_3).
    # Two ranks contain NaNs (bad data); the other two are clean.
    rank_arrays = {
        "rank_0": [1.0, 2.0, 3.0, 4.0],
        "rank_1": [3.0, 6.0, float("nan"), 12.0],           # has NaNs
        #"rank_2": [0.5, 1.5, 2.5, 3.5],
        #"rank_3": [5.0, float("nan"), 3.0, float("nan")],   # has NaNs
    }
    for _key, _arr in rank_arrays.items():
        data_store[_key] = _arr
    data_keys = list(rank_arrays)

    num_ranks = len(data_keys)
    registry = ToolRegistry()
    # Single-call tool: checks every rank in one shot and returns a ready-made
    # report.  This keeps the agent's job to "call once, relay the report",
    # which small models perform far more reliably than an N-step tool loop.
    registry.register(make_scan_all_ranks(data_store, num_ranks))

    print("[startup] Launching lightweight inference service...", flush=True)
    inference_proc = Process(
        target=lightweight_inference_service,
        args=(input_queue, inference_shutdown, LOCAL_MODEL_DIR),
    )
    inference_proc.start()

    procs, agent_specs = [], []
    try:
        # --- Create the single agent ---
        agent_spec = {
            "config": AgentConfig(
                agent_id="nan_agent",
                name="NaN Detector",
                role=(
                    "You are a data-quality assistant.  You call the "
                    "scan_all_ranks tool exactly once and report its 'report' "
                    "string verbatim as your answer."
                ),
                inference_queue=input_queue,
                # Single agent, one request at a time — no need for the
                # proxy's default pool of 32 response channels.
                max_concurrent_requests=1,
                # One tool call + one final answer is all that is needed now,
                # but leave headroom in case the model retries the tool once.
                max_tool_call_iterations=6,
            ),
            "tool_registry": registry,
            "shutdown_event": Event(),
            "reply_queue": Queue(),
        }
        agent_specs = [agent_spec]

        # Launch the agent as a Dragon Process.
        p = Process(target=create_sub_agent, kwargs=agent_spec)
        p.start()
        procs.append(p)

        # The agent publishes its own input queue once it is ready.
        agent_input_queue = agent_spec["reply_queue"].get()
        agent_spec["config"].input_queue = agent_input_queue
        print("[startup] Agent 'nan_agent' ready.\n", flush=True)

        orchestrator = DAGOrchestrator(
            config=OrchestratorConfig(
                agents=[agent_spec["config"]],
                poll_interval=0.5,
                poll_timeout=1200.0,
            ),
            pipeline=pipeline,
        )

        user_input = (
            f"Result arrays from {num_ranks} MPI ranks are stored in the "
            "shared data store.  Call the scan_all_ranks tool once, then "
            "report its 'report' string as your answer."
        )

        batch = Batch(results_ddict_mem=int(10*1024*1024))
        try:
            print("=" * 60, flush=True)
            print("Dragon AI — NaN Detector (lightweight inference)", flush=True)
            print("=" * 60, flush=True)
            print(f"Request: {user_input}\n", flush=True)

            result = orchestrator.run(user_input=user_input, batch=batch)

            # orchestrator.run() returns the terminal node's result dict,
            # shaped like {"response": <final answer text>}.  Pull the text
            # out so we print the agent's actual words, not the dict repr.
            if isinstance(result, dict):
                final_text = result.get("response") or result.get("error") or ""
            else:
                final_text = str(result)

            print("\n" + "=" * 60, flush=True)
            print("FINAL RESULT (agent's answer)", flush=True)
            print("=" * 60, flush=True)
            print(final_text.strip() or "[agent returned no final text]", flush=True)

            # Deterministic cross-check straight from the shared store, so the
            # keys with NaNs are always reported even if the model's final
            # text comes back weak or empty.
            print("\n--- Keys with NaNs (ground truth) ---", flush=True)
            any_flagged = False
            for k in data_keys:
                check = _check_nans(data_store[k])
                if check["has_nans"]:
                    any_flagged = True
                    print(
                        f"  {k}: {check['nan_count']} NaN(s) at "
                        f"indices {check['nan_indices']}",
                        flush=True,
                    )
            if not any_flagged:
                print("  none", flush=True)
        except Exception as exc:
            import traceback
            print(f"\n[error] Pipeline failed: {exc}", flush=True)
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
    asyncio.run(main())
