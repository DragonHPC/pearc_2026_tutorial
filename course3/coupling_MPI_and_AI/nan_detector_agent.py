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

            # The agent's DragonQueueLLMProxy accepts a dict with "assistant".
            response_queue.put({"assistant": text})
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

def ultralight_inference_service(input_queue, shutdown_event, model_name):
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
            "You are a data-quality assistant.  The user tells you the DDict "
            "key under which a list of numbers is stored.  Determine whether "
            "that list contains any NaN (missing/invalid) values by calling "
            "the detect_nans tool with that key (do NOT pass the numbers "
            "themselves — pass only the key string).\n\n"
            "After the tool returns, state clearly whether NaNs are present, "
            "how many there are, and at which indices."
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
    # Create a small DDict (1 manager, 1 node, 2 MB) and stage the dataset
    # under a key.  The payload here is tiny (~80 bytes), so there is no need
    # to reserve a large managed pool.  The agent passes this key to the
    # detect_nans tool instead of the raw list.
    # See develop/examples/dragon_data/ddict/demo_ddict.py.
    data_store = DDict(1, 1, 2 * 1024 * 1024)
    data_key = "sensor_readings"
    data_store[data_key] = [1.0, 2.5, float("nan"), 4.2, 5.0, float("nan"), 7.7]

    registry = ToolRegistry()
    registry.register(make_detect_nans(data_store))

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
                    "You are a data-quality assistant.  Detect NaN values in "
                    "numeric lists by calling detect_nans, then summarise the "
                    "findings for the user."
                ),
                inference_queue=input_queue,
                # Single agent, one request at a time — no need for the
                # proxy's default pool of 32 response channels.
                max_concurrent_requests=1,
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
                poll_timeout=600.0,
            ),
            pipeline=pipeline,
        )

        user_input = (
            f"A list of sensor readings is stored in the shared data store "
            f"under the key '{data_key}'. "
            "Do any of them contain NaN values?"
        )

        batch = Batch(results_ddict_mem=int(10*1024*1024))
        try:
            print("=" * 60, flush=True)
            print("Dragon AI — NaN Detector (lightweight inference)", flush=True)
            print("=" * 60, flush=True)
            print(f"Request: {user_input}\n", flush=True)

            result = orchestrator.run(user_input=user_input, batch=batch)

            print("\n" + "=" * 60, flush=True)
            print("FINAL RESULT", flush=True)
            print("=" * 60, flush=True)
            print(result, flush=True)
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
