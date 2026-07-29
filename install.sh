#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "    Installing Air-Gapped Local MLX Agent Harness     "
echo "========================================================"

setup_kaggle_credentials() {
  if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
    echo "[*] Using Kaggle credentials from environment variables."
    return 0
  fi

  if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "[*] Using Kaggle credentials from $HOME/.kaggle/kaggle.json"
    return 0
  fi

  echo ""
  echo "--------------------------------------------------------"
  echo " [!] Kaggle Authentication Required for Model Downloads "
  echo "--------------------------------------------------------"
  echo "Get your API key at: https://www.kaggle.com/settings -> Account -> Create New Token"
  echo ""
  
  read -r -p "Enter your Kaggle Username: " kaggle_user
  read -r -s -p "Enter your Kaggle API Key: " kaggle_key
  echo ""

  if [ -z "$kaggle_user" ] || [ -z "$kaggle_key" ]; then
    echo "[!] Warning: Kaggle credentials skipped. Download may fail if models require auth."
    return 0
  fi

  mkdir -p "$HOME/.kaggle"
  (
    umask 077
    cat <<EOF > "$HOME/.kaggle/kaggle.json"
{
  "username": "$kaggle_user",
  "key": "$kaggle_key"
}
EOF
  )
  chmod 600 "$HOME/.kaggle/kaggle.json"
  echo "[+] Configured $HOME/.kaggle/kaggle.json (chmod 600)"

  export KAGGLE_USERNAME="$kaggle_user"
  export KAGGLE_KEY="$kaggle_key"
}

echo ""
echo "[1/7] Setting up KaggleHub credentials..."
setup_kaggle_credentials

echo ""
echo "[2/7] Installing Python dependencies (requirements.txt)..."
if command -v pip3 &>/dev/null; then
  pip3 install -r requirements.txt
elif command -v pip &>/dev/null; then
  pip install -r requirements.txt
else
  echo "[!] Error: pip/pip3 not found." >&2
  exit 1
fi

echo ""
echo "[3/7] Downloading KaggleHub model assets..."
python3 scripts/download_models.py

echo ""
echo "[4/7] Building local BGE vector RAG index..."
python3 scripts/build_vector_index.py

echo ""
echo "[5/7] Installing Node dependencies & running postinstall model detector..."
npm install

echo ""
echo "[6/7] Building workspace Node packages..."
npm run build

echo ""
echo "[7/7] Verifying repository integrity & updating exports..."
npm run check

# Automatically update environment exports and source env.sh
node scripts/detect_models.mjs
if [ -f "env.sh" ]; then
  # Sourcing env.sh into current shell process
  set -a
  # shellcheck disable=SC1091
  source "env.sh"
  set +a
  echo "[+] Automatically sourced env.sh into execution environment."
fi

echo ""
echo "========================================================"
echo "           Installation Complete Successfully           "
echo "========================================================"
echo ""
echo "Environment variables exported:"
echo "  OPENCODE_MLX_MODEL_PATH=\"${OPENCODE_MLX_MODEL_PATH:-}\""
echo "  OPENCODE_BGE_EMBED_PATH=\"${OPENCODE_BGE_EMBED_PATH:-}\""
echo "  OPENCODE_VECTORS_FILE=\"${OPENCODE_VECTORS_FILE:-}\""
echo "  OPENCODE_METADATA_FILE=\"${OPENCODE_METADATA_FILE:-}\""
echo ""
echo "To start pi in local offline MLX mode:"
echo "  source env.sh   # (or: source install.sh)"
echo "  ./pi-test.sh --provider stdio-local --model local-llm --offline"
echo ""
