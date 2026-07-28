#!/usr/bin/env python3
import os
import sys
import json
import re
import numpy as np

# --- Security & Corporate Air-Gapped Enforcements ---
os.environ["HF_HUB_OFFLINE"] = "1"

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_embeddings.utils import load as load_embeddings

# --- Configuration Paths ---
MODEL_PATH = os.environ.get("OPENCODE_MLX_MODEL_PATH", "/Users/a819340/Downloads/MLX/mlx_model/")
EMBED_PATH = os.environ.get("OPENCODE_BGE_EMBED_PATH", "/Users/a819340/.cache/kagglehub/models/jonathanchan/baai/transformers/bge-large-en-v1.5/1/")
VECTORS_FILE = os.environ.get("OPENCODE_VECTORS_FILE", "/Users/a819340/Downloads/MLX/code_vectors.npy")
METADATA_FILE = os.environ.get("OPENCODE_METADATA_FILE", "/Users/a819340/Downloads/MLX/code_metadata.json")

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
    input_ids = embed_tokenizer.encode(query, max_length=512, truncation=True, return_tensors="np")
    outputs = embed_model(mx.array(input_ids))
    query_vector = outputs.text_embeds.astype(mx.float32)
    query_vector = query_vector / mx.linalg.norm(query_vector, axis=-1, keepdims=True)
    scores = mx.matmul(code_embeddings, query_vector.T).flatten()
    mx.eval(scores)
    top_indices = mx.argsort(-scores)[:top_k].tolist()
    context = "\n=== RELEVANT CODEBASE FILES ===\n"
    for idx in top_indices:
        match = code_vault[idx]
        context += f"\n[FILE: {match['file']}]\n{match['text']}\n"
    return context


# --- Prompt Construction with Tools ---
def format_tools_manual(tools: list) -> str:
    """Fallback tool formatter for models without native chat-template tool support."""
    if not tools:
        return ""
    lines = [
        "\n# Tools",
        "You may call one or more tools by writing a <tool_call> block:",
        "<tool_call>",
        '{"name": "TOOL_NAME", "arguments": {"arg1": "value1"}}',
        "</tool_call>",
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

    parts = []
    system = next((m.get("content") for m in messages if m.get("role") == "system"), "")
    if tools:
        system += format_tools_manual(tools)
    if system:
        parts.append(f"<start_of_turn>system\n{system}<end_of_turn>")

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
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == "assistant":
            tcs = msg.get("tool_calls", [])
            if tcs:
                for tc in tcs:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    content += f"\n<tool_call>\n{{\"name\": \"{fn.get('name')}\", \"arguments\": {args}}}\n</tool_call>"
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        elif role == "tool":
            parts.append(f"<start_of_turn>tool\n{content}<end_of_turn>")

    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


# --- Response Parsing ---
TOOL_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)

def parse_response(full_text: str):
    """Split response into (clean_text, tool_calls[])."""
    tool_calls = []
    for i, match in enumerate(TOOL_RE.findall(full_text)):
        try:
            data = json.loads(match)
            name = data.get("name") or data.get("function", {}).get("name")
            args = data.get("arguments") or data.get("function", {}).get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append({
                "id": f"call_{i:03d}",
                "type": "function",
                "function": {"name": name, "arguments": args}
            })
        except (json.JSONDecodeError, KeyError):
            continue

    clean = TOOL_RE.sub("", full_text).strip()
    return clean, tool_calls


# --- Main IPC Loop ---
print("[*] ENGINE_READY", file=sys.stderr)
sys.stderr.flush()

while True:
    line = sys.stdin.readline()
    if not line:
        break

    try:
        req = json.loads(line)

        messages = req.get("messages", [])
        tools = req.get("tools", [])
        max_tokens = req.get("max_tokens", 8192)

        # Inject RAG context into the last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                raw_content = messages[i].get("content", "")
                if isinstance(raw_content, list):
                    user_str = "\n".join(c.get("text", "") for c in raw_content if c.get("type") == "text")
                else:
                    user_str = str(raw_content)

                ctx = local_context_search(user_str)
                if ctx:
                    messages[i]["content"] = ctx + "\n\n" + user_str
                break

        prompt_str = build_prompt(tokenizer, messages, tools)

        full_response = ""
        text_acc = ""
        in_tool = False
        tool_buffer = ""

        for chunk in stream_generate(model, tokenizer, prompt=prompt_str, max_tokens=max_tokens):
            token_text = chunk.text
            full_response += token_text

            if not in_tool:
                if "<tool_call>" in full_response:
                    idx = full_response.index("<tool_call>")
                    before = full_response[:idx]
                    new_text = before[len(text_acc):]
                    if new_text:
                        sys.stdout.write(json.dumps({"type": "text", "text": new_text}) + "\n")
                        sys.stdout.flush()
                    text_acc = before
                    in_tool = True
                    tool_buffer = full_response[idx:]
                else:
                    sys.stdout.write(json.dumps({"type": "text", "text": token_text}) + "\n")
                    sys.stdout.flush()
                    text_acc = full_response
            else:
                tool_buffer += token_text
                if "</tool_call>" in tool_buffer:
                    end_idx = tool_buffer.index("</tool_call>") + len("</tool_call>")
                    complete_block = tool_buffer[:end_idx]
                    remainder = tool_buffer[end_idx:]
                    full_response = text_acc + complete_block + remainder

                    _, tcs = parse_response(complete_block)
                    for tc in tcs:
                        sys.stdout.write(json.dumps({
                            "type": "tool_call",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "arguments": json.dumps(tc["function"]["arguments"])
                        }) + "\n")
                        sys.stdout.flush()

                    in_tool = False
                    tool_buffer = ""
                    text_acc = full_response

                    if remainder:
                        sys.stdout.write(json.dumps({"type": "text", "text": remainder}) + "\n")
                        sys.stdout.flush()

            eos_ids = [tokenizer.eos_token_id]
            if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
                eos_ids.append(tokenizer.pad_token_id)
            if chunk.token in eos_ids:
                break

        if not in_tool and len(full_response) > len(text_acc):
            remainder = full_response[len(text_acc):]
            if remainder:
                sys.stdout.write(json.dumps({"type": "text", "text": remainder}) + "\n")
                sys.stdout.flush()

        clean_text, tool_calls = parse_response(full_response)
        for tc in tool_calls:
            sys.stdout.write(json.dumps({
                "type": "tool_call",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": json.dumps(tc["function"]["arguments"])
            }) + "\n")
            sys.stdout.flush()

        sys.stdout.write(json.dumps({"type": "done"}) + "\n")
        sys.stdout.flush()

    except Exception as e:
        sys.stdout.write(json.dumps({"type": "error", "error": str(e)}) + "\n")
        sys.stdout.flush()
