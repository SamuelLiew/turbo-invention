# Pi Agent Harness (Local Air-Gapped MLX Fork)

This is a local, air-gapped fork of the Pi agent harness powered by an Apple Silicon GPU-accelerated **MLX Engine**, **BGE Embedding Vector RAG**, and **`stdio-local` process transport**.

## Key Features

* **`stdio-local` Transport**: Native process communication via `stdio-openai` and `stdio-local` providers without external network requests.
* **Apple Silicon MLX Engine** (`scripts/mlx_engine.py`): Metal GPU-accelerated local LLM inference via `mlx-lm`.
* **Codebase BGE RAG Search**: In-memory vector matrix search using `bge-large-en-v1.5` embeddings for context retrieval.
* **Native Tool Calling**: Protocol support for streaming `<tool_call>` events, enabling Pi to execute local workspace tools (`read`, `write`, `edit`, `grep`, `find`, `ls`, etc.) in multi-turn loops.
* **Air-Gapped Privacy & Lockdown**: Remote cloud provider connections, external telemetry, auto-downloads, and version checks are disabled.

---

## Quickstart & Setup

### 1. Kaggle Authentication & Model Downloads
Configure Kaggle API credentials and download local model weights (Gemma GGUF + BGE embeddings):

```bash
./scripts/setup_kaggle_models.sh
```

### 2. Python Dependencies
Ensure required MLX and RAG Python packages are installed:

```bash
pip install -r requirements.txt
```

### 3. Running Pi with Local Stdio Provider

Run Pi locally with the `stdio-local` provider:

```bash
./pi-test.sh --provider stdio-local --model local-llm --offline
```

---

## Project Structure & Packages

| Package / Directory | Description |
|---------------------|-------------|
| **[@earendil-works/pi-ai](packages/ai)** | Unified LLM API with `stdio-openai` / `stdio-local` transport |
| **[@earendil-works/pi-agent-core](packages/agent)** | Agent execution runtime, state management, and tool call handling |
| **[@earendil-works/pi-coding-agent](packages/coding-agent)** | Interactive CLI coding agent |
| **[@earendil-works/pi-tui](packages/tui)** | Terminal UI library |
| **`scripts/mlx_engine.py`** | Local MLX LLM engine + BGE vector search process |
| **`scripts/download_models.py`** | KaggleHub asset downloader |
| **`scripts/setup_kaggle_models.sh`** | Setup helper for Kaggle credentials and dependencies |

---

## Local Development & Testing

```bash
npm install --ignore-scripts  # Install dependencies without running lifecycle scripts
npm run build:offline         # Offline build without network fetching
npm run check                 # Code formatting, linting, and type checking
./test.sh                     # Non-e2e test suite
./pi-test.sh                  # Launch Pi from local source files
```

---

## License

MIT
