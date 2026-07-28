#!/usr/bin/env python3
import sys

try:
    import kagglehub
except ImportError:
    print("[!] kagglehub is not installed. Run 'pip install kagglehub' first.", file=sys.stderr)
    sys.exit(1)

print("========================================================")
print("  Downloading KaggleHub Models for Local Execution      ")
print("========================================================")

print("\n[*] [1/3] Downloading BGE Embedding Model (bge-large-en-v1.5)...")
path1 = kagglehub.model_download("jonathanchan/baai/transformers/bge-large-en-v1.5")
print(f" -> Path to BGE Embedding model files: {path1}")

print("\n[*] [2/3] Downloading BGE Reranker Model (bge-reranker-large)...")
path2 = kagglehub.model_download("jonathanchan/baai/transformers/bge-reranker-large")
print(f" -> Path to BGE Reranker model files: {path2}")

print("\n[*] [3/3] Downloading Local Language Model (gemma-4-e4b-it-qat-q4_0-gguf)...")
path3 = kagglehub.model_download("google/gemma-4/gguf/gemma-4-e4b-it-qat-q4_0-gguf")
print(f" -> Path to Gemma 4 model files: {path3}")

print("\n[+] All model files downloaded successfully!")
