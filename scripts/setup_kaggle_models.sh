#!/usr/bin/env bash
set -euo pipefail

# --- Step 0. Kaggle Credentials Setup ---
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
  cat <<EOF > "$HOME/.kaggle/kaggle.json"
{
  "username": "$kaggle_user",
  "key": "$kaggle_key"
}
EOF
  chmod 600 "$HOME/.kaggle/kaggle.json"
  echo "[+] Configured $HOME/.kaggle/kaggle.json (chmod 600)"

  ZSHRC="$HOME/.zshrc"
  if [ -f "$ZSHRC" ] || [ "${SHELL:-}" = "*/zsh" ] || [ -n "${ZSH_VERSION:-}" ] || [ "$(uname)" = "Darwin" ]; then
    touch "$ZSHRC"
    if ! grep -q "KAGGLE_USERNAME" "$ZSHRC" 2>/dev/null; then
      echo "" >> "$ZSHRC"
      echo "# Kaggle API Credentials" >> "$ZSHRC"
      echo "export KAGGLE_USERNAME=\"$kaggle_user\"" >> "$ZSHRC"
      echo "export KAGGLE_KEY=\"$kaggle_key\"" >> "$ZSHRC"
      echo "[+] Appended KAGGLE_USERNAME and KAGGLE_KEY to $ZSHRC"
    fi
  fi

  export KAGGLE_USERNAME="$kaggle_user"
  export KAGGLE_KEY="$kaggle_key"
}

setup_kaggle_credentials

echo ""
echo "[1/2] Installing Python MLX, BGE & KaggleHub dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/2] Downloading local model assets via KaggleHub..."
python3 scripts/download_models.py
