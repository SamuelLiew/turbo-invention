# Pi Agent Harness (Local Air-Gapped MLX Fork)

This is an air-gapped, privacy-focused fork of the Pi coding agent harness. It is powered by an Apple Silicon GPU-accelerated **MLX Engine**, **BGE Embedding Vector RAG**, and **`stdio-local` process transport**.

---

## System Requirements

* **OS**: macOS (Apple Silicon M1/M2/M3/M4 recommended for Metal GPU acceleration)
* **Node.js**: >= 22.19.0
* **Python**: >= 3.10

---

## Detailed Step-by-Step Usage Guide

### Step 1: Install Dependencies

1. **Install Node.js Monorepo Dependencies**:
   ```bash
   npm install --ignore-scripts
   ```

2. **Install Python MLX & BGE RAG Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

### Step 2: Set Up Kaggle Credentials & Download Models

Download the required local model weights (Gemma 4 GGUF + BGE embeddings):

```bash
./scripts/setup_kaggle_models.sh
```

> **Note**: This interactive script configures `~/.kaggle/kaggle.json` (chmod `600`) and downloads:
> - `jonathanchan/baai/transformers/bge-large-en-v1.5` (Embedding Model)
> - `jonathanchan/baai/transformers/bge-reranker-large` (Reranker Model)
> - `google/gemma-4/gguf/gemma-4-e4b-it-qat-q4_0-gguf` (Local Model)

---

### Step 3: Configure Environment Variables (Optional)

If your model weights or BGE vector indices are stored in custom directories, export these variables before running:

```bash
export OPENCODE_MLX_MODEL_PATH="/path/to/mlx_model/"
export OPENCODE_BGE_EMBED_PATH="$HOME/.cache/kagglehub/models/jonathanchan/baai/transformers/bge-large-en-v1.5/1/"
export OPENCODE_VECTORS_FILE="/path/to/code_vectors.npy"
export OPENCODE_METADATA_FILE="/path/to/code_metadata.json"
```

---

### Step 4: Run Pi Agent

#### 1. Interactive Terminal Mode (TUI)
Launch the interactive coding session using `stdio-local`:

```bash
./pi-test.sh --provider stdio-local --model local-llm --offline
```

#### 2. Restricted Safe Tools Mode (Recommended for Air-Gap Security)
Run without `bash` process execution capabilities:

```bash
./pi-test.sh --provider stdio-local --model local-llm --offline --tools read,edit,write,grep,find,ls
```

#### 3. Single-Prompt Mode
Execute a single coding task or inquiry:

```bash
./pi-test.sh --provider stdio-local --model local-llm --offline -p "Analyze the architecture of packages/ai"
```

---

## Troubleshooting & Tips

* **`ENGINE_READY` Check**: The MLX Python process emits `[*] ENGINE_READY` on `stderr` once the Metal GPU weights and BGE vector index are loaded into memory.
* **Persistent Process Execution**: The Python engine runs as a long-running IPC process, ensuring model weights remain warm in Mac GPU memory across multi-turn agent conversations.
* **Air-Gap Verification**: All remote cloud provider factories, automatic version checking, and telemetry are disabled.

---

## Directory Overview

| File / Package | Description |
|----------------|-------------|
| **`scripts/mlx_engine.py`** | Metal GPU MLX engine + BGE vector search process |
| **`scripts/download_models.py`** | KaggleHub model asset downloader |
| **`scripts/setup_kaggle_models.sh`** | Interactive Kaggle credentials and model download helper |
| **`requirements.txt`** | Python dependencies (`mlx`, `mlx-lm`, `mlx-embeddings`, `numpy`, `kagglehub`) |
| **[@earendil-works/pi-ai](packages/ai)** | LLM API with `stdio-local` transport |
| **[@earendil-works/pi-coding-agent](packages/coding-agent)** | Interactive CLI coding agent harness |

---

## License

MIT
