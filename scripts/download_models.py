#!/usr/bin/env python3
import os
import sys

try:
    import kagglehub
except ImportError:
    print("[!] kagglehub is not installed. Run 'pip install -r requirements.txt' first.", file=sys.stderr)
    sys.exit(1)

try:
    from mlx_lm import load
except ImportError:
    print("[!] mlx_lm is not installed. Run 'pip install -r requirements.txt' first.", file=sys.stderr)
    sys.exit(1)

print("========================================================")
print("  Downloading Models for Local MLX Execution           ")
print("========================================================")

print("\n[*] [1/3] Downloading BGE Embedding Model (bge-large-en-v1.5)...")
path1 = kagglehub.model_download("jonathanchan/baai/transformers/bge-large-en-v1.5")
print(f" -> Path to BGE Embedding model files: {path1}")

print("\n[*] [2/3] Downloading BGE Reranker Model (bge-reranker-large)...")
path2 = kagglehub.model_download("jonathanchan/baai/transformers/bge-reranker-large")
print(f" -> Path to BGE Reranker model files: {path2}")

print("\n[*] [3/3] Downloading Local Language Model Transformers")
path3 = kagglehub.model_download("coolgamerz/qwen3-6-27b-4bitmlx/other/default")
print(f" -> Path to Gemma 4 model files: {path3}")

print("\n[+] All model assets downloaded & verified successfully!")
