from dragon.native.queue import Queue
from dragon.native.event import Event
from dragon.ai.inference.inference_utils import Inference
from dragon.ai.inference.config import (
    InferenceConfig, ModelConfig, HardwareConfig, BatchingConfig,
    GuardrailsConfig, DynamicWorkerConfig,
)
import argparse
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional
import os
import queue
from dragon.infrastructure.policy import Policy
import dragon.ai.torch
import torch
import threading
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

# Use Dragon as multiprocessineg backend. Safe for reruns in notebooks.


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_DIR = _REPO_ROOT / "model"
LOCAL_MODEL_DIR = os.environ.get(
    "DRAGON_LOCAL_MODEL_DIR",
    str(_DEFAULT_MODEL_DIR if _DEFAULT_MODEL_DIR.exists() else Path("/tmp/model")),
)

def setup_inference_service():

    inference_queue = Queue()

    inference = Inference(
        config=InferenceConfig(
            backend="cpu",
            model=ModelConfig(
                model_name=str(LOCAL_MODEL_DIR),
                hf_token="",
                tp_size=1,
                max_tokens=16,
                max_model_len=256,
                dtype="float16",
            ),
            hardware=HardwareConfig(
                # CPU-only execution
                num_gpus=-1,
                node_offset=0,
                num_nodes=1,
            ),
            guardrails=GuardrailsConfig(enabled=False),
            dynamic_worker=DynamicWorkerConfig(enabled=False),
            flask_secret_key="",
            run_type="agent",
            token="",
        ),
        input_queue=inference_queue,
    )
    inference.initialize()
    return inference, inference_queue

def run_hello_world_test(prompt: str, timeout_seconds: float = 30.0) -> int:
    response_queue = Queue()
    input_queue = Queue()
    shutdown_event = Event()

    input_queue.put((response_queue, prompt))
    inference_service_light(input_queue, shutdown_event=shutdown_event)

    #response = _queue_get_with_timeout(response_queue, timeout_seconds)
    print("getting response",flush=True)
    response = response_queue.get()
    print("got response",flush=True)
    shutdown_event.set()
    if response is None:
        print(
            "[agents.py] No response within timeout. "
            "Inference started, but request/response queue schema may differ in this environment."
        )
        return 1

    if isinstance(response, (dict, list)):
        print("[agents.py] Model response:")
        print(json.dumps(response, indent=2, default=str))
    else:
        print("[agents.py] Model response:")
        print(str(response))

    return 0

def inference_service_light(
        prompt_queue,
        *,
        model_name="../../model",
        device="cuda" if torch.cuda.is_available() else "cpu",
        shutdown_event=None,
):
    "Reads prompts from a queue, runs inference, places response from LLM into response queue."


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    ).to(device)
    print("compiled model", flush=True)
    model.eval()

    #while shutdown_event is None or not shutdown_event.is_set():
    count = 0
    while count == 0:
        try:
            # If no more work after some timeout, cleanly exit.
            response_queue, prompt = prompt_queue.get(timeout=1)
        except queue.Empty:
            continue
        except Exception as e:
            return True

        # Prepare input for model.
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        max_new_tokens = 64
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate the output.
        start = time.time()
        print("generating response",flush=True)
        with torch.inference_mode():
            generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        # Get and decode the output.
        print(f"generated response in {time.time() - start}",flush=True)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        print("putting response",flush=True)
        response_queue.put(tokenizer.decode(output_ids, skip_special_tokens=True))
        print("put response",flush=True)
        count +=1

def main() -> int:
    parser = argparse.ArgumentParser(description="Run a hello-world test against Dragon LLM inference.")
    parser.add_argument(
        "--prompt",
        default="Hello world! Please reply with one short sentence.",
        help="Prompt to send to the local model.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for a response before timing out.",
    )
    args = parser.parse_args()

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    print("Dragon + PyTorch ready")
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    print("PyTorch threads:", torch.get_num_threads())

    try:
        return run_hello_world_test(prompt=args.prompt, timeout_seconds=args.timeout)
    except Exception as exc:
        print(f"[agents.py] Test failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())