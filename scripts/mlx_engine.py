#!/usr/bin/env python3
"""
turbo-invention MLX Inference Engine
Fixed version addressing all critical issues from code review.

Protocol: NDJSON over stdin/stdout
  TypeScript -> Python: {"request_id":"...","messages":[...],"tools":[...],...}
  Python -> TypeScript: {"request_id":"...","type":"text|tool_call|done|error",...}
"""

import os
import sys
import json
import re
import copy
import itertools

# --- Security & Corporate Air-Gapped Enforcements ---
os.environ["HF_HUB_OFFLINE"] = "1"

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_embeddings.utils import load as load_embeddings

# --- Configuration Paths (no hardcoded usernames) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.environ.get(
    "OPENCODE_MLX_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "mlx_model")
)
EMBED_PATH = os.environ.get(
    "OPENCODE_BGE_EMBED_PATH",
    os.path.join(PROJECT_ROOT, "models", "bge-large-en-v1.5")
)
VECTORS_FILE = os.environ.get(
    "OPENCODE_VECTORS_FILE",
    os.path.join(PROJECT_ROOT, "data", "code_vectors.npy")
)
METADATA_FILE = os.environ.get(
    "OPENCODE_METADATA_FILE",
    os.path.join(PROJECT_ROOT, "data", "code_metadata.json")
)

# --- RAG Limits ---
MAX_RAG_CHARS_PER_FILE = 4000
MAX_RAG_TOTAL_CHARS = 12000

print("[*] Initializing GPU MLX Engine with Tool Calling + BGE RAG...", file=sys.stderr)

try:
    model, tokenizer = load(MODEL_PATH)
    embed_model, embed_tokenizer = load_embeddings(EMBED_PATH)

    if os.path.exists(VECTORS_FILE) and os.path.exists(METADATA_FILE):
        raw_embeddings = np.load(VECTORS_FILE)
        code_embeddings = mx.array(raw_embeddings, mx.float32)
        code_embeddings = code_embeddings / mx.linalg.norm(code_embeddings, axis=-1, keepdims=True)
        with open(METADATA_FILE, "r") as f:
            code_vault = json.load(f)
        print(f"[+] Loaded {len(code_vault)} vectors into Mac GPU memory.", file=sys.stderr)
    else:
        code_embeddings, code_vault = None, None
        print("[!] Warning: Vector index files missing.", file=sys.stderr)
except Exception as e:
    print(f"[!] Initialization Failure: {e}", file=sys.stderr)
    sys.exit(1)


# --- RAG Search ---
def local_context_search(query: str, top_k: int = 3) -> str:
    if code_embeddings is None or code_vault is None:
        return ""

    # FIX: Handle both dict and array returns from tokenizer.encode
    encoded = embed_tokenizer.encode(
        query, max_length=512, truncation=True, return_tensors="np"
    )
    if isinstance(encoded, dict):
        input_ids = mx.array(encoded["input_ids"])
    else:
        input_ids = mx.array(encoded)

    outputs = embed_model(input_ids)

    # FIX: Defensive attribute access for different embedding model APIs
    try:
        query_vector = outputs.text_embeds.astype(mx.float32)
    except AttributeError:
        query_vector = outputs.astype(mx.float32)

    query_vector = query_vector / mx.linalg.norm(query_vector, axis=-1, keepdims=True)
    scores = mx.matmul(code_embeddings, query_vector.T).flatten()
    mx.eval(scores)
    top_indices = mx.argsort(-scores)[:top_k].tolist()

    context = "\n=== RELEVANT CODEBASE FILES ===\n"
    total_chars = len(context)

    for idx in top_indices:
        match = code_vault[idx]
        file_text = match.get("text", "")

        # FIX: Enforce per-file and total context limits
        if len(file_text) > MAX_RAG_CHARS_PER_FILE:
            file_text = file_text[:MAX_RAG_CHARS_PER_FILE] + "\n... [truncated]"

        entry = f"\n[FILE: {match.get('file', 'unknown')}]\n{file_text}\n"
        if total_chars + len(entry) > MAX_RAG_TOTAL_CHARS:
            break

        context += entry
        total_chars += len(entry)

    return context


# --- Prompt Construction with Tools ---
def format_tools_manual(tools: list) -> str:
    """Fallback tool formatter for models without native chat-template tool support."""
    if not tools:
        return ""
    lines = [
        "\n# Tools",
        "You may call one or more tools by writing a <tool_call> block:",
        "",
        '<tool_call>{"name": "TOOL_NAME", "arguments": {"arg1": "value1"}}</tool_call>',
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
    """Convert OpenAI-style messages to model prompt string."""
    # Prefer native chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tools=tools or None,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception as e:
            print(f"[!] apply_chat_template failed ({e}), using manual fallback.", file=sys.stderr)

    # --- Manual Fallback (Gemma-compatible) ---
    # FIX: Use proper <start_of_turn> / <end_of_turn> delimiters for Gemma
    parts = []
    system_content = ""

    system = next((m.get("content") for m in messages if m.get("role") == "system"), "")
    if tools:
        system += format_tools_manual(tools)
    if system:
        system_content = system

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
                # FIX: Wrap historical tool calls in <tool_call> tags
                for tc in tcs:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    content += f'\n<tool_call>{{"name": "{fn.get("name")}", "arguments": {args}}}</tool_call>'
            parts.append(("model", content))
        elif role == "tool":
            tc_id = msg.get("tool_call_id", "")
            prefix = f"[TOOL_RESULT id={tc_id}]\n" if tc_id else ""
            parts.append(("user", f"{prefix}{content}"))

    # Prepend system to first user message (Gemma convention)
    if system_content and parts:
        first_role, first_content = parts[0]
        if first_role == "user":
            parts[0] = ("user", f"{system_content}\n\n{first_content}")
        else:
            parts.insert(0, ("user", system_content))
    elif system_content:
        parts.insert(0, ("user", system_content))

    prompt_lines = []
    for role, content in parts:
        prompt_lines.append(f"<start_of_turn>{role}\n{content}<end_of_turn>")

    prompt_lines.append("<start_of_turn>model\n")
    return "\n".join(prompt_lines)


# --- Response Parsing (State Machine, not Regex) ---
def extract_tool_calls(text: str):
    """
    Extract tool calls from text using a proper scanner.
    Returns: (clean_text: str, tool_calls: list, error: str|None)
    """
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
            # Unclosed tag - include rest as text and warn
            clean_parts.append(text[start:])
            return "".join(clean_parts).strip(), tool_calls, "Unclosed <tool_call> tag"

        json_str = text[start + len("<tool_call>"):end].strip()
        try:
            data = json.loads(json_str)
            name = data.get("name") or data.get("function", {}).get("name")
            args = data.get("arguments") or data.get("function", {}).get("arguments", {})
            if name is None:
                raise KeyError("Missing 'name' in tool_call")
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append({
                "id": f"call_{len(tool_calls):03d}",
                "type": "function",
                "function": {"name": name, "arguments": args}
            })
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # FIX: Return error instead of silently swallowing
            return None, None, f"Malformed tool_call JSON: {e}. Raw: {json_str[:200]}"

        i = end + len("</tool_call>")

    return "".join(clean_parts).strip(), tool_calls, None


# --- Request Validation ---
def validate_request(req: dict) -> tuple[bool, str]:
    if not isinstance(req, dict):
        return False, "Request must be a JSON object"
    if "messages" not in req:
        return False, "Missing 'messages' field"
    if not isinstance(req.get("messages"), list):
        return False, "'messages' must be an array"
    return True, ""


# --- Main IPC Loop ---
print("[*] ENGINE_READY", file=sys.stderr)
sys.stderr.flush()

while True:
    line = sys.stdin.readline()
    if not line:
        break

    try:
        req = json.loads(line)
    except json.JSONDecodeError as e:
        sys.stdout.write(json.dumps({"type": "error", "error": f"Invalid JSON: {e}"}) + "\n")
        sys.stdout.flush()
        continue

    request_id = req.get("request_id", "")

    ok, err_msg = validate_request(req)
    if not ok:
        sys.stdout.write(json.dumps({"type": "error", "error": err_msg, "request_id": request_id}) + "\n")
        sys.stdout.flush()
        continue

    try:
        # FIX: Deep-copy messages to prevent mutating shared references from TypeScript
        messages = copy.deepcopy(req.get("messages", []))
        tools = req.get("tools", [])
        max_tokens = req.get("max_tokens", 8192)
        temp = req.get("temperature", 0.7)

        # Inject RAG context into the last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                raw_content = messages[i].get("content", "")
                if isinstance(raw_content, list):
                    user_str = "\n".join(
                        c.get("text", "") for c in raw_content if c.get("type") == "text"
                    )
                else:
                    user_str = str(raw_content)

                ctx = local_context_search(user_str)
                if ctx:
                    messages[i]["content"] = ctx + "\n\n" + user_str
                break

        prompt_str = build_prompt(tokenizer, messages, tools)

        # FIX: Build sampling params cleanly, avoid massive code duplication
        generate_kwargs = {"prompt": prompt_str, "max_tokens": max_tokens}
        sampling_params = {}
        if temp is not None:
            sampling_params["temp"] = temp
        if req.get("top_p") is not None:
            sampling_params["top_p"] = req["top_p"]
        if req.get("repetition_penalty") is not None:
            sampling_params["repetition_penalty"] = req["repetition_penalty"]

        def _stream_with_sampling():
            return stream_generate(model, tokenizer, **{**generate_kwargs, **sampling_params})

        # Try with sampling params; fallback once if unsupported
        try:
            gen_iter = _stream_with_sampling()
            # Peek to validate params are accepted
            first_chunk = next(gen_iter)
            gen_iter = itertools.chain([first_chunk], gen_iter)
        except TypeError:
            gen_iter = stream_generate(model, tokenizer, **generate_kwargs)

        # --- Unified Streaming Handler ---
        buffer = ""
        processed_pos = 0
        in_tool = False
        tool_start_pos = 0
        eos_ids = [tokenizer.eos_token_id]
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            eos_ids.append(tokenizer.pad_token_id)

        for chunk in gen_iter:
            token_text = chunk.text
            buffer += token_text

            # Check for EOS
            if hasattr(chunk, "token") and chunk.token in eos_ids:
                break

            while True:
                if not in_tool:
                    tag_start = buffer.find("<tool_call>", processed_pos)
                    if tag_start == -1:
                        # Emit all new text as normal
                        new_text = buffer[processed_pos:]
                        if new_text:
                            sys.stdout.write(json.dumps({
                                "type": "text",
                                "text": new_text,
                                "request_id": request_id
                            }) + "\n")
                            sys.stdout.flush()
                        processed_pos = len(buffer)
                        break
                    else:
                        # Emit text before tool call
                        text_before = buffer[processed_pos:tag_start]
                        if text_before:
                            sys.stdout.write(json.dumps({
                                "type": "text",
                                "text": text_before,
                                "request_id": request_id
                            }) + "\n")
                            sys.stdout.flush()
                        in_tool = True
                        tool_start_pos = tag_start
                        processed_pos = tag_start
                else:
                    tag_end = buffer.find("</tool_call>", tool_start_pos)
                    if tag_end == -1:
                        # Still inside tool call, wait for more tokens
                        break

                    complete_block = buffer[tool_start_pos:tag_end + len("</tool_call>")]
                    clean, tcs, err = extract_tool_calls(complete_block)

                    if err:
                        sys.stdout.write(json.dumps({
                            "type": "error",
                            "error": err,
                            "request_id": request_id
                        }) + "\n")
                        sys.stdout.flush()
                        raise RuntimeError(err)

                    for tc in tcs:
                        sys.stdout.write(json.dumps({
                            "type": "tool_call",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "arguments": json.dumps(tc["function"]["arguments"]),
                            "request_id": request_id
                        }) + "\n")
                        sys.stdout.flush()

                    in_tool = False
                    processed_pos = tag_end + len("</tool_call>")
                    # Continue loop to check for more tool calls or trailing text

        # Emit any remaining text after streaming ends
        if not in_tool and processed_pos < len(buffer):
            remainder = buffer[processed_pos:]
            if remainder:
                sys.stdout.write(json.dumps({
                    "type": "text",
                    "text": remainder,
                    "request_id": request_id
                }) + "\n")
                sys.stdout.flush()

        # Final safety parse (catches any tool calls missed by streaming, e.g. if EOS hit mid-tag)
        if in_tool:
            # Unclosed tag at end of generation
            sys.stdout.write(json.dumps({
                "type": "error",
                "error": "Model ended generation inside a <tool_call> tag",
                "request_id": request_id
            }) + "\n")
            sys.stdout.flush()
        else:
            clean_text, final_tcs, err = extract_tool_calls(buffer)
            if err:
                sys.stdout.write(json.dumps({
                    "type": "error",
                    "error": err,
                    "request_id": request_id
                }) + "\n")
                sys.stdout.flush()
            elif final_tcs and not in_tool:
                # Emit any tool calls that weren't caught during streaming
                # (shouldn't happen with the state machine, but defensive)
                for tc in final_tcs:
                    sys.stdout.write(json.dumps({
                        "type": "tool_call",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": json.dumps(tc["function"]["arguments"]),
                        "request_id": request_id
                    }) + "\n")
                    sys.stdout.flush()

        # Token usage estimation
        prompt_tokens = len(tokenizer.encode(prompt_str)) if hasattr(tokenizer, "encode") else 0
        completion_tokens = len(tokenizer.encode(buffer)) if hasattr(tokenizer, "encode") else 0

        sys.stdout.write(json.dumps({
            "type": "done",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "request_id": request_id
        }) + "\n")
        sys.stdout.flush()

    except Exception as e:
        sys.stdout.write(json.dumps({
            "type": "error",
            "error": str(e),
            "request_id": request_id
        }) + "\n")
        sys.stdout.flush()