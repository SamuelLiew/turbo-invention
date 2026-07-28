#!/usr/bin/env python3
import os
import sys
import ssl
import json
import numpy as np

# --- Security & Corporate Air-Gapped Enforcements ---
os.environ["HF_HUB_OFFLINE"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_embeddings.utils import load as load_embeddings

# --- Configuration Paths ---
MODEL_PATH = os.environ.get("OPENCODE_MLX_MODEL_PATH", "/Users/a819340/Downloads/MLX/mlx_model/")
EMBED_PATH = os.environ.get("OPENCODE_BGE_EMBED_PATH", "/Users/a819340/.cache/kagglehub/models/jonathanchan/baai/transformers/bge-large-en-v1.5/1/")
VECTORS_FILE = os.environ.get("OPENCODE_VECTORS_FILE", "/Users/a819340/Downloads/MLX/code_vectors.npy")
METADATA_FILE = os.environ.get("OPENCODE_METADATA_FILE", "/Users/a819340/Downloads/MLX/code_metadata.json")

print("[*] Initializing GPU MLX Engine and Pre-Normalized BGE Vector Matrix...", file=sys.stderr)

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

def local_context_search(query, top_k=3):
    if code_embeddings is None or code_vault is None:
        return ""
    input_ids = embed_tokenizer.encode(query, max_length=512, truncation=True, return_tensors="mlx")
    outputs = embed_model(input_ids)
    query_vector = outputs.text_embeds.astype(mx.float32)
    query_vector = query_vector / mx.linalg.norm(query_vector, axis=-1, keepdims=True)
    scores = mx.matmul(code_embeddings, query_vector.T).flatten()
    mx.eval(scores)
    top_indices = mx.argsort(scores)[::-1][:top_k].tolist()
    context = "\n=== RELEVANT CODEBASE FILES ===\n"
    for idx in top_indices:
        match = code_vault[idx]
        context += f"\n[FILE: {match['file']}]\n{match['text']}\n"
    return context

print("[*] ENGINE_READY", file=sys.stderr)
sys.stderr.flush()

# --- Main stdin / stdout IPC Protocol Loop ---
while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        req = json.loads(line)
        user_query = req.get("prompt", "")
        max_tokens = req.get("max_tokens", 8192)

        # Retrieve relevant code context via GPU matrix multiplication
        extracted_context = local_context_search(user_query)
        final_prompt = f"{extracted_context}\n\nQuestion: {user_query}" if extracted_context else user_query
        prompt_str = f"<|im_start|>user\n{final_prompt}<|im_end|>\n<|im_start|>assistant\n"

        in_thought = False
        full_response = ""

        for chunk in stream_generate(model, tokenizer, prompt=prompt_str, max_tokens=max_tokens):
            text_fragment = chunk.text
            eos_str = tokenizer.eos_token if isinstance(tokenizer.eos_token, str) else "<|im_end|>"
            
            if any(meta in text_fragment for meta in ["<|im_end|>", "<|endoftext|>", eos_str]):
                break
                
            full_response += text_fragment
            
            # Detect thinking tags
            if not in_thought and any(tag in full_response[-30:] for tag in ["<|channel>thought", "<thought>", "<think>"]):
                in_thought = True
                continue
            if in_thought and any(tag in full_response[-30:] for tag in ["<channel|>", "<|channel>content", "</thought>", "</think>"]):
                in_thought = False
                continue

            clean_fragment = text_fragment
            for tag in ["<|channel>thought", "<|channel>content", "<|channel>", "<channel|>", "<thought>", "</thought>", "<think>", "</think>"]:
                clean_fragment = clean_fragment.replace(tag, "")

            if clean_fragment:
                # Stream NDJSON object per token over stdout
                event = {"type": "thought" if in_thought else "content", "text": clean_fragment}
                sys.stdout.write(json.dumps(event) + "\n")
                sys.stdout.flush()

        sys.stdout.write(json.dumps({"type": "done"}) + "\n")
        sys.stdout.flush()

    except Exception as e:
        sys.stdout.write(json.dumps({"type": "error", "error": str(e)}) + "\n")
        sys.stdout.flush()
