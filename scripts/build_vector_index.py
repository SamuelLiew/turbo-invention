#!/usr/bin/env python3
"""
Build the BGE vector index consumed by scripts/mlx_engine.py.

Walks a source tree, embeds each chunk with the local BGE model and writes the
two files the engine loads at startup:
  - code_vectors.npy   (float32 matrix, one row per chunk)
  - code_metadata.json (list of {"file", "chunk", "text"} aligned with the rows)

Usage:
  python3 scripts/build_vector_index.py [--root .] [--out-dir data]
"""

import argparse
import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import mlx.core as mx
from mlx_embeddings.utils import load as load_embeddings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

EMBED_PATH = os.environ.get(
    "OPENCODE_BGE_EMBED_PATH",
    os.path.join(PROJECT_ROOT, "models", "bge-large-en-v1.5")
)

# Keep chunks under the BGE 512-token window with room to spare.
CHUNK_CHARS = 1500
MAX_FILE_BYTES = 512 * 1024
INDEXED_SUFFIXES = {
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".sh", ".json",
    ".md", ".toml", ".yml", ".yaml",
}
SKIP_DIRS = {
    ".git", "node_modules", "dist", "build", "coverage", "models", "data",
    ".venv", "__pycache__", ".artifacts",
}


def iter_source_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if os.path.splitext(name)[1] not in INDEXED_SUFFIXES:
                continue
            full = os.path.join(dirpath, name)
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield full


def chunk_text(text: str):
    return [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)] or [""]


def embed(model, tokenizer, text: str):
    encoded = tokenizer.encode(text, max_length=512, truncation=True, return_tensors="np")
    input_ids = mx.array(encoded["input_ids"] if isinstance(encoded, dict) else encoded)
    outputs = model(input_ids)
    try:
        vector = outputs.text_embeds.astype(mx.float32)
    except AttributeError:
        vector = outputs.astype(mx.float32)
    mx.eval(vector)
    return np.array(vector, dtype=np.float32).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the BGE code vector index.")
    parser.add_argument("--root", default=PROJECT_ROOT, help="Tree to index (default: repo root)")
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "data"),
                        help="Where to write code_vectors.npy / code_metadata.json")
    args = parser.parse_args()

    print(f"[*] Loading BGE embedding model from {EMBED_PATH}", file=sys.stderr)
    model, tokenizer = load_embeddings(EMBED_PATH)

    vectors, metadata = [], []
    for path in iter_source_files(args.root):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            print(f"[!] Skipping {path}: {e}", file=sys.stderr)
            continue

        rel = os.path.relpath(path, args.root)
        for index, chunk in enumerate(chunk_text(content)):
            vectors.append(embed(model, tokenizer, chunk))
            metadata.append({"file": rel, "chunk": index, "text": chunk})

        print(f"[+] {rel}", file=sys.stderr)

    if not vectors:
        print("[!] No indexable files found.", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    vectors_file = os.path.join(args.out_dir, "code_vectors.npy")
    metadata_file = os.path.join(args.out_dir, "code_metadata.json")

    np.save(vectors_file, np.stack(vectors))
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    print(f"[+] Wrote {len(vectors)} vectors to {vectors_file} and {metadata_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
