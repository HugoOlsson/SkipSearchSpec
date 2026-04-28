#!/usr/bin/env bash
#
# Automated setup script for SkipSearchSpec
# Usage: chmod +x setup.sh && ./setup.sh
#

set -euo pipefail

# Use sudo only if not already root
if [[ $EUID -ne 0 ]]; then
    SUDO="sudo"
else
    SUDO=""
fi

echo "==> [1/5] Updating apt and installing rsync, tmux, curl, python3, python3-venv, git..."
$SUDO apt-get update
$SUDO apt-get install -y rsync tmux curl python3 python3-venv git

echo "==> [2/5] Installing Poetry..."
if ! command -v poetry >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/poetry" ]]; then
    curl -sSL https://install.python-poetry.org | python3 -
else
    echo "    Poetry already installed, skipping."
fi

# Make poetry available in this shell
export PATH="$HOME/.local/bin:$PATH"

# Persist PATH for future shells (only if not already there)
if ! grep -qs 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    echo "    Added ~/.local/bin to PATH in ~/.bashrc"
fi

echo "==> [3/5] Cloning SkipSearchSpec repository..."
if [[ -d "SkipSearchSpec/.git" ]]; then
    echo "    Repository already cloned, pulling latest..."
    git -C SkipSearchSpec pull --ff-only
else
    git clone https://github.com/HugoOlsson/SkipSearchSpec.git
fi

cd SkipSearchSpec

echo "==> [4/5] Making install_server.sh executable..."
chmod +x scripts/install_server.sh

echo "==> [5/5] Running install_server.sh..."
./scripts/install_server.sh

echo ""
echo "==> Setup complete."
echo "    PATH was updated in ~/.bashrc — open a new shell or run:"
echo "        source ~/.bashrc"


echo "==> [6/6] Configuring Hugging Face token..."
if [[ -z "${HF_TOKEN:-}" ]]; then
    read -rsp "    Enter Hugging Face token (input hidden, leave blank to skip): " HF_TOKEN
    echo
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    mkdir -p "$HOME/.cache/huggingface"
    printf '%s' "$HF_TOKEN" > "$HOME/.cache/huggingface/token"
    chmod 600 "$HOME/.cache/huggingface/token"

    # Persist for future shells (idempotent)
    if ! grep -qs '^export HF_TOKEN=' "$HOME/.bashrc" 2>/dev/null; then
        echo "export HF_TOKEN='$HF_TOKEN'" >> "$HOME/.bashrc"
    fi
    export HF_TOKEN
    echo "    HF token written to ~/.cache/huggingface/token and ~/.bashrc"
else
    echo "    No HF_TOKEN provided, skipping. You can set it later with:"
    echo "        export HF_TOKEN=hf_xxx"
fi