#!/usr/bin/env bash
#
# RunPod startup script for SkipSearchSpec.
#
# Put this script in the RunPod template startup command, or run:
#
#   bash /workspace/SkipSearchSpec/scripts/runpod_startup.sh
#
# The script is intentionally idempotent. System packages still need to be
# installed on each fresh container, but the repository, Poetry, virtualenv,
# Hugging Face assets, Torch caches, and other large downloads live on the
# persistent RunPod volume under /workspace.

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
PROJECT_NAME="${PROJECT_NAME:-SkipSearchSpec}"
PROJECT_ROOT="${PROJECT_ROOT:-$WORKSPACE_DIR/$PROJECT_NAME}"
REPO_URL="${REPO_URL:-https://github.com/HugoOlsson/SkipSearchSpec.git}"
REPO_BRANCH="${REPO_BRANCH:-}"
RUNPOD_UPDATE_REPO="${RUNPOD_UPDATE_REPO:-0}"
RUNPOD_RUN_LOCK="${RUNPOD_RUN_LOCK:-1}"
RUNPOD_FORCE_INSTALL="${RUNPOD_FORCE_INSTALL:-0}"

export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKSPACE_DIR/.cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$WORKSPACE_DIR/.config}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$WORKSPACE_DIR/.local/share}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
export POETRY_HOME="${POETRY_HOME:-$WORKSPACE_DIR/.poetry}"
export POETRY_CACHE_DIR="${POETRY_CACHE_DIR:-$XDG_CACHE_HOME/pypoetry}"
export POETRY_CONFIG_DIR="${POETRY_CONFIG_DIR:-$XDG_CONFIG_HOME/pypoetry}"
export POETRY_VIRTUALENVS_IN_PROJECT=true
export POETRY_NO_INTERACTION=1

export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-$WORKSPACE_DIR/wandb}"
export TMPDIR="${TMPDIR:-$WORKSPACE_DIR/tmp}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PATH="$POETRY_HOME/bin:$HOME/.local/bin:$PATH"

stage() {
  printf '\n==> %s\n' "$*"
}

append_once() {
  local line="$1"
  local file="$2"

  mkdir -p "$(dirname "$file")"
  touch "$file"
  if ! grep -Fqx "$line" "$file"; then
    printf '%s\n' "$line" >> "$file"
  fi
}

link_if_absent() {
  local target="$1"
  local link="$2"

  mkdir -p "$target" "$(dirname "$link")"
  if [[ ! -e "$link" && ! -L "$link" ]]; then
    ln -s "$target" "$link"
  fi
}

install_system_packages() {
  stage "Installing container packages"

  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    apt-get update
    apt-get install -y rsync tmux curl git python3 python3-venv
  else
    sudo apt-get update
    sudo apt-get install -y rsync tmux curl git python3 python3-venv
  fi
}

install_poetry() {
  stage "Installing Poetry on the persistent volume"

  if [[ ! -x "$POETRY_HOME/bin/poetry" ]]; then
    curl -fsSL https://install.python-poetry.org | python3 -
    hash -r
  else
    "$POETRY_HOME/bin/poetry" --version
  fi
}

clone_or_update_repo() {
  stage "Preparing repository at $PROJECT_ROOT"

  mkdir -p "$WORKSPACE_DIR"

  if [[ ! -d "$PROJECT_ROOT/.git" ]]; then
    local clone_args=()
    if [[ -n "$REPO_BRANCH" ]]; then
      clone_args+=(--branch "$REPO_BRANCH")
    fi
    git clone "${clone_args[@]}" "$REPO_URL" "$PROJECT_ROOT"
    return
  fi

  cd "$PROJECT_ROOT"

  if [[ "$RUNPOD_UPDATE_REPO" != "1" ]]; then
    echo "Repository already exists. Set RUNPOD_UPDATE_REPO=1 to fetch/pull on startup."
    return
  fi

  git fetch origin

  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Local changes detected; leaving repository untouched."
    return
  fi

  local upstream
  if upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)"; then
    git pull --ff-only
  else
    echo "No upstream configured for current branch; fetched origin but did not pull."
  fi
}

write_shell_environment() {
  stage "Writing persistent shell environment"

  mkdir -p \
    "$XDG_CACHE_HOME" \
    "$XDG_CONFIG_HOME" \
    "$XDG_DATA_HOME" \
    "$PIP_CACHE_DIR" \
    "$POETRY_HOME" \
    "$POETRY_CACHE_DIR" \
    "$POETRY_CONFIG_DIR" \
    "$HF_HOME" \
    "$HF_HUB_CACHE" \
    "$HF_DATASETS_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$TORCH_HOME" \
    "$TRITON_CACHE_DIR" \
    "$MPLCONFIGDIR" \
    "$WANDB_DIR" \
    "$TMPDIR"

  cat > "$WORKSPACE_DIR/skipsearchspec_env.sh" <<EOF
export WORKSPACE_DIR="$WORKSPACE_DIR"
export PROJECT_ROOT="$PROJECT_ROOT"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
export XDG_CONFIG_HOME="$XDG_CONFIG_HOME"
export XDG_DATA_HOME="$XDG_DATA_HOME"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export POETRY_HOME="$POETRY_HOME"
export POETRY_CACHE_DIR="$POETRY_CACHE_DIR"
export POETRY_CONFIG_DIR="$POETRY_CONFIG_DIR"
export POETRY_VIRTUALENVS_IN_PROJECT=true
export POETRY_NO_INTERACTION=1
export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export TORCH_HOME="$TORCH_HOME"
export TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export MPLCONFIGDIR="$MPLCONFIGDIR"
export WANDB_DIR="$WANDB_DIR"
export TMPDIR="$TMPDIR"
export PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
export TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM"
export PATH="$POETRY_HOME/bin:\$HOME/.local/bin:\$PATH"
EOF

  append_once 'source /workspace/skipsearchspec_env.sh' "$HOME/.bashrc"
  append_once 'source /workspace/skipsearchspec_env.sh' "$HOME/.zshrc"

  link_if_absent "$HF_HOME" "$HOME/.cache/huggingface"
  link_if_absent "$TORCH_HOME" "$HOME/.cache/torch"
  link_if_absent "$PIP_CACHE_DIR" "$HOME/.cache/pip"

  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s' "$HF_TOKEN" > "$HF_HOME/token"
    chmod 600 "$HF_HOME/token"
  fi
}

configure_project() {
  stage "Configuring Poetry project"

  cd "$PROJECT_ROOT"

  poetry config virtualenvs.in-project true
  poetry config cache-dir "$POETRY_CACHE_DIR"

  if [[ "$RUNPOD_RUN_LOCK" == "1" ]]; then
    poetry lock --no-interaction
  else
    echo "Skipping poetry lock because RUNPOD_RUN_LOCK=$RUNPOD_RUN_LOCK"
  fi
}

server_environment_is_ready() {
  if [[ "$RUNPOD_FORCE_INSTALL" == "1" ]]; then
    return 1
  fi

  if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    return 1
  fi

  cd "$PROJECT_ROOT"

  poetry run python - <<'PY'
import importlib.util
import sys

try:
    import torch
except Exception as exc:
    print(f"torch import failed: {exc}")
    raise SystemExit(1)

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.version.cuda != "12.8":
    raise SystemExit(1)

if not torch.cuda.is_available():
    raise SystemExit(1)

missing = [
    name
    for name in ("fla", "fla.ops", "fla.layers", "causal_conv1d")
    if importlib.util.find_spec(name) is None
]

if missing:
    print("missing:", ", ".join(missing))
    raise SystemExit(1)

raise SystemExit(0)
PY
}

install_python_environment() {
  stage "Installing Python project environment"

  cd "$PROJECT_ROOT"

  if server_environment_is_ready; then
    echo "Existing /workspace virtualenv already has the server fast-path dependencies."
    poetry install --with build -E server
    return
  fi

  bash ./scripts/install_server.sh
}

print_summary() {
  stage "RunPod setup complete"

  cat <<EOF
Project root:       $PROJECT_ROOT
Virtualenv:         $PROJECT_ROOT/.venv
Poetry:             $POETRY_HOME
Poetry cache:       $POETRY_CACHE_DIR
pip cache:          $PIP_CACHE_DIR
Hugging Face home:  $HF_HOME
HF hub cache:       $HF_HUB_CACHE
HF datasets cache:  $HF_DATASETS_CACHE
Torch cache:        $TORCH_HOME
Triton cache:       $TRITON_CACHE_DIR

Start working with:
  cd "$PROJECT_ROOT"
  source "$WORKSPACE_DIR/skipsearchspec_env.sh"
  poetry run python -m skip_search_spec.main
EOF
}

main() {
  install_system_packages
  write_shell_environment
  install_poetry
  clone_or_update_repo
  configure_project
  install_python_environment
  print_summary
}

main "$@"
