#!/usr/bin/env python3
"""
Qwen3 MLX Inference Engine — OpenAI-compatible stdio protocol for pi
"""

import os
import sys
import time
import json
import copy
import uuid
import inspect

os.environ["HF_HUB_OFFLINE"] = "1"
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

import numpy as np
import mlx.core as mx
from mlx_lm import load, stream_generate

# --- Hardcoded to your Kaggle cache ---
MODEL_PATH = os.environ.get(
    "OPENCODE_MLX_MODEL_PATH",
    "/Users/coda/.cache/kagglehub/models/coolgamerz/qwen3-6-27b-4bitmlx/other/default/1"
)

# --- Optional BGE / RAG ---
EMBED_PATH = os.environ.get("OPENCODE_BGE_EMBED_PATH", "")
VECTORS_FILE = os.environ.get("OPENCODE_VECTORS_FILE", "")
METADATA_FILE = os.environ.get("OPENCODE_METADATA_FILE", "")

MAX_RAG_CHARS_PER_FILE = 4000
MAX_RAG_TOTAL_CHARS = 12000
BGE_QUERY_PREFIX = os.environ.get(
    "OPENCODE_BGE_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: "
)
RAG_MIN_SCORE = float(os.environ.get("OPENCODE_RAG_MIN_SCORE", "0.3"))

print(f"[*] Initializing MLX Engine with model: {MODEL_PATH}", file=sys.stderr)
sys.stderr.flush()

config_path = os.path.join(MODEL_PATH, "config.json")
if not os.path.exists(config_path):
    err = f"config.json not found at {MODEL_PATH}"
    print(json.dumps({"error": {"message": err, "type": "engine_error"}}), file=sys.stdout)
    sys.stdout.flush()
    sys.exit(1)

try:
    t0 = time.time()
    model, tokenizer = load(MODEL_PATH)
    print(f"[*] Loaded LLM in {time.time()-t0:.2f}s", file=sys.stderr)
    print(f"[*] Default MLX device: {mx.default_device()}", file=sys.stderr)
    sys.stderr.flush()
except Exception as e:
    err = f"Failed to load model: {e}"
    print(json.dumps({"error": {"message": err, "type": "engine_error"}}), file=sys.stdout)
    sys.stdout.flush()
    sys.exit(1)

# --- Optional embeddings ---
embed_model = None
embed_tokenizer = None
code_embeddings = None
code_vault = None

if EMBED_PATH and os.path.exists(os.path.join(EMBED_PATH, "config.json")):
    try:
        from mlx_embeddings.utils import load as load_embeddings
        t0 = time.time()
        embed_model, embed_tokenizer = load_embeddings(EMBED_PATH)
        print(f"[*] Loaded BGE in {time.time()-t0:.2f}s", file=sys.stderr)
    except Exception as e:
        print(f"[!] BGE load failed (RAG disabled): {e}", file=sys.stderr)

if embed_model is not None and VECTORS_FILE and os.path.exists(VECTORS_FILE) and os.path.exists(METADATA_FILE):
    try:
        t0 = time.time()
        raw = np.load(VECTORS_FILE)
        code_embeddings = mx.array(raw, mx.float32)
        code_embeddings = code_embeddings / mx.linalg.norm(code_embeddings, axis=-1, keepdims=True)
        with open(METADATA_FILE) as f:
            code_vault = json.load(f)
        print(f"[*] Loaded {len(code_vault)} vectors in {time.time()-t0:.2f}s", file=sys.stderr)
    except Exception as e:
        print(f"[!] Vector load failed (RAG disabled): {e}", file=sys.stderr)
        code_embeddings = None
        code_vault = None
else:
    print("[*] RAG disabled (no embeddings/vectors found)", file=sys.stderr)

sys.stderr.flush()


def local_context_search(query: str, top_k: int = 3) -> str:
    if code_embeddings is None or code_vault is None or embed_model is None:
        return ""
    try:
        encoded = embed_tokenizer.encode(
            BGE_QUERY_PREFIX + query, max_length=512, truncation=True, return_tensors="np"
        )
        input_ids = mx.array(encoded["input_ids"] if isinstance(encoded, dict) else encoded)
        outputs = embed_model(input_ids)
        try:
            query_vector = outputs.text_embeds.astype(mx.float32)
        except AttributeError:
            query_vector = outputs.astype(mx.float32)
        query_vector = query_vector / mx.linalg.norm(query_vector, axis=-1, keepdims=True)
        scores = mx.matmul(code_embeddings, query_vector.T).flatten()
        mx.eval(scores)
        top_indices = mx.argsort(-scores)[:top_k].tolist()
        score_list = scores.tolist()

        header = "\n=== RELEVANT CODEBASE FILES ===\n"
        context = ""
        total_chars = len(header)
        matches = 0
        for idx in top_indices:
            if score_list[idx] < RAG_MIN_SCORE:
                continue
            match = code_vault[idx]
            text = match.get("text", "")
            if len(text) > MAX_RAG_CHARS_PER_FILE:
                text = text[:MAX_RAG_CHARS_PER_FILE] + "\n... [truncated]"
            entry = f"\n[FILE: {match.get('file', 'unknown')}]\n{text}\n"
            if total_chars + len(entry) > MAX_RAG_TOTAL_CHARS:
                break
            context += entry
            total_chars += len(entry)
            matches += 1
        return header + context if context else ""
    except Exception as e:
        print(f"[!] RAG search error: {e}", file=sys.stderr)
        return ""


def format_tools_manual(tools: list) -> str:
    if not tools:
        return ""
    lines = [
        "\n# Tools",
        "You may call one or more tools by writing a <tool_call> block:",
        "",
        '{"name": "TOOL_NAME", "arguments": {"arg1": "value1"}}',
        "",
        "\nAvailable tools:"
    ]
    for tool in tools:
        fn = tool.get("function", {})
        lines.append(f"\n- {fn.get('name')}: {fn.get('description', '')}")
        lines.append(f"  Parameters: {json.dumps(fn.get('parameters', {}))}")
        lines.append("\nIf no tool is needed, respond normally.")
    return "\n".join(lines)


def build_prompt(tokenizer, messages: list, tools: list | None) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tools=tools or None, add_generation_prompt=True, tokenize=False
            )
        except Exception as e:
            print(f"[!] apply_chat_template failed ({e}), using manual fallback.", file=sys.stderr)

    # --- Manual Qwen3 fallback ---
    parts = []
    system = next((m.get("content") for m in messages if m.get("role") == "system"), "")
    if tools:
        system += format_tools_manual(tools)

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
        elif not isinstance(content, str):
            content = str(content)
        if role == "system":
            continue
        elif role == "user":
            parts.append(("user", content))
        elif role == "assistant":
            tcs = msg.get("tool_calls", [])
            if tcs:
                for tc in tcs:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    content += f'\n<tool_call>{{"name": "{fn.get("name")}", "arguments": {args}}}</tool_call>'
            parts.append(("assistant", content))
        elif role == "tool":
            tc_id = msg.get("tool_call_id", "")
            prefix = f"[TOOL_RESULT id={tc_id}]\n" if tc_id else ""
            parts.append(("user", f"{prefix}{content}"))

    if system and parts:
        if parts[0][0] == "user":
            parts[0] = ("user", f"{system}\n\n{parts[0][1]}")
        else:
            parts.insert(0, ("user", system))
    elif system:
        parts.insert(0, ("user", system))

    prompt_lines = [f"<|im_start|>{role}\n{content}<|im_end|>" for role, content in parts]
    prompt_lines.append("<|im_start|>assistant\n")
    return "\n".join(prompt_lines)


def extract_tool_calls(text: str, id_offset: int = 0):
    tool_calls = []
    clean_parts = []
    i = 0
    while True:
        start = text.find("<tool_call>", i)
        if start == -1:
            clean_parts.append(text[i:])
            break
        clean_parts.append(text[i:start])
        end = text.find("</tool_call>", start)
        if end == -1:
            clean_parts.append(text[start:])
            return "".join(clean_parts).strip(), tool_calls, "Unclosed <tool_call> tag"
        json_str = text[start + len("<tool_call>"):end].strip()
        try:
            data = json.loads(json_str)
            name = data.get("name") or data.get("function", {}).get("name")
            args = data.get("arguments") or data.get("function", {}).get("arguments", {})
            if name is None:
                raise KeyError("Missing 'name'")
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append({
                "id": f"call_{id_offset + len(tool_calls):03d}",
                "type": "function",
                "function": {"name": name, "arguments": args}
            })
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return None, None, f"Malformed tool_call JSON: {e}. Raw: {json_str[:200]}"
        i = end + len("</tool_call>")
    return "".join(clean_parts).strip(), tool_calls, None


# --- Sampler compatibility layer ---
def _accepted_kwargs(fn):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return set(params)

def _resolve_kwarg(fn, canonical: str) -> str | None:
    accepted = _accepted_kwargs(fn)
    for name in (canonical,):
        if accepted is None or name in accepted:
            return name
    return None

def build_sampler(temp, top_p):
    """Build a sampler compatible with the installed mlx-lm version."""
    try:
        from mlx_lm import sample_utils
    except ImportError:
        sample_utils = None

    make_sampler = getattr(sample_utils, "make_sampler", None)
    if make_sampler is not None:
        sampler_kwargs = {}
        if temp is not None:
            resolved = _resolve_kwarg(make_sampler, "temp")
            if resolved:
                sampler_kwargs[resolved] = temp
            else:
                print("[!] make_sampler does not support 'temp'", file=sys.stderr)
        if top_p is not None:
            resolved = _resolve_kwarg(make_sampler, "top_p")
            if resolved:
                sampler_kwargs[resolved] = top_p
            else:
                print("[!] make_sampler does not support 'top_p'", file=sys.stderr)
        if sampler_kwargs:
            return make_sampler(**sampler_kwargs)
    return None

def build_generate_kwargs(prompt: str, max_tokens: int, temp, top_p, repetition_penalty=None):
    kwargs = {"prompt": prompt, "max_tokens": max_tokens}

    sampler = build_sampler(temp, top_p)
    if sampler is not None and _resolve_kwarg(stream_generate, "sampler"):
        kwargs["sampler"] = sampler
    else:
        # Fall back to direct kwargs on stream_generate
        if temp is not None:
            resolved = _resolve_kwarg(stream_generate, "temp")
            if resolved:
                kwargs[resolved] = temp
            else:
                print("[!] stream_generate does not support 'temp'; ignoring.", file=sys.stderr)
        if top_p is not None:
            resolved = _resolve_kwarg(stream_generate, "top_p")
            if resolved:
                kwargs[resolved] = top_p
            else:
                print("[!] stream_generate does not support 'top_p'; ignoring.", file=sys.stderr)

    if repetition_penalty is not None:
        try:
            from mlx_lm import sample_utils
            make_logits_processors = getattr(sample_utils, "make_logits_processors", None)
        except ImportError:
            make_logits_processors = None

        if make_logits_processors is not None and _resolve_kwarg(stream_generate, "logits_processors"):
            kwargs["logits_processors"] = make_logits_processors(repetition_penalty=repetition_penalty)
        elif _resolve_kwarg(stream_generate, "repetition_penalty"):
            kwargs["repetition_penalty"] = repetition_penalty
        else:
            print("[!] repetition_penalty not supported; ignoring.", file=sys.stderr)

    return kwargs


def _send_chunk(content: str = "", tool_calls=None, finish_reason=None, model_id="local-llm"):
    delta = {}
    if content:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls

    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    }
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()


# --- Main IPC loop ---
print("[*] ENGINE_READY", file=sys.stderr)
sys.stderr.flush()

while True:
    line = sys.stdin.readline()
    if not line:
        break

    try:
        req = json.loads(line)
    except json.JSONDecodeError as e:
        _send_chunk(finish_reason="stop")
        continue

    # OpenAI-compatible request fields
    messages = req.get("messages", [])
    tools = req.get("tools", [])
    model_id = req.get("model", "local-llm")
    max_tokens = req.get("max_tokens", 8192)
    temp = req.get("temperature", 0.7)
    top_p = req.get("top_p", 1.0)
    repetition_penalty = req.get("repetition_penalty")

    if not messages:
        _send_chunk(finish_reason="stop", model_id=model_id)
        continue

    try:
        # --- RAG injection ---
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                raw = messages[i].get("content", "")
                if isinstance(raw, list):
                    user_str = "\n".join(c.get("text", "") for c in raw if c.get("type") == "text")
                else:
                    user_str = str(raw)
                ctx = local_context_search(user_str)
                if ctx:
                    messages[i]["content"] = ctx + "\n\n" + user_str
                break

        prompt_str = build_prompt(tokenizer, messages, tools)

        # --- Generation ---
        generate_kwargs = build_generate_kwargs(prompt_str, max_tokens, temp, top_p, repetition_penalty)

        t_gen_start = time.time()
        t_first_token = None
        gen_iter = stream_generate(model, tokenizer, **generate_kwargs)

        buffer = ""
        processed_pos = 0
        emitted_tool_calls = 0
        eos_ids = [tokenizer.eos_token_id]
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            eos_ids.append(tokenizer.pad_token_id)

        for chunk in gen_iter:
            if t_first_token is None:
                t_first_token = time.time()
                print(f"[*] TTFT: {(t_first_token - t_gen_start) * 1000:.1f}ms", file=sys.stderr)
                sys.stderr.flush()
                # OpenAI protocol: emit empty delta with role first
                _send_chunk(content="", model_id=model_id)

            token_text = chunk.text
            buffer += token_text

            if hasattr(chunk, "token") and chunk.token in eos_ids:
                break

            # Stream word-by-word to balance latency vs throughput
            last_space = buffer.rfind(' ', processed_pos)
            if last_space != -1 and last_space > processed_pos:
                word = buffer[processed_pos:last_space + 1]
                _send_chunk(content=word, model_id=model_id)
                processed_pos = last_space + 1

        # --- Flush remainder ---
        if processed_pos < len(buffer):
            tail = buffer[processed_pos:]
            clean_text, tcs, err = extract_tool_calls(tail, emitted_tool_calls)
            if err:
                print(f"[!] Tool parse error: {err}", file=sys.stderr)
                clean_text = tail
            if tcs:
                for tc in tcs:
                    _send_chunk(
                        tool_calls=[{
                            "index": 0,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": json.dumps(tc["function"]["arguments"])
                            }
                        }],
                        model_id=model_id
                    )
                emitted_tool_calls += len(tcs)
            if clean_text:
                _send_chunk(content=clean_text, model_id=model_id)

        # --- Final chunk ---
        _send_chunk(finish_reason="stop", model_id=model_id)

        prompt_tokens = len(tokenizer.encode(prompt_str)) if hasattr(tokenizer, "encode") else 0
        completion_tokens = len(tokenizer.encode(buffer)) if hasattr(tokenizer, "encode") else 0
        gen_duration = time.time() - t_gen_start
        t_speed = completion_tokens / gen_duration if gen_duration > 0 else 0.0
        print(f"[*] Generation: {completion_tokens} tokens in {gen_duration:.2f}s ({t_speed:.1f} tok/s)", file=sys.stderr)
        sys.stderr.flush()

    except Exception as e:
        print(f"[!] Generation error: {e}", file=sys.stderr)
        sys.stderr.flush()
        _send_chunk(content=f"[Engine error: {e}]", finish_reason="stop", model_id=model_id)