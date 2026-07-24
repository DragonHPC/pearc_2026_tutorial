import os
import queue as _pyqueue

MAX_NEW_TOKENS = int(os.environ.get("DRAGON_MAX_NEW_TOKENS", "1024"))


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

def _parse_cfls(text: str) -> list:
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
    vals = [v for v in pool if 0 < v <= 10]

    return [round(v, 4) for v in vals]



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
            #print(
            #    f"[inference][debug] schema_hint={'yes' if len(request) > 5 and request[5] else 'no'} "
            #    f"tools={'yes' if tools else 'no'} raw_output={text!r}",
            #    flush=True,
            #)

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

            print(f"[inference] payload={payload!r}", flush=True)
            # The agent's DragonQueueLLMProxy accepts a dict with "assistant".
            response_queue.put({"assistant": payload})
        except Exception as exc:  # noqa: BLE001 — report failure to the agent
            # Returning the exception lets the proxy re-raise it agent-side.
            response_queue.put(exc)

    print("[inference] Shutdown signalled — inference service stopping.", flush=True)
