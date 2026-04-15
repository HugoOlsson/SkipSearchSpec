#!/bin/bash
#SBATCH -J sss-early-exit
#SBATCH -p long
#SBATCH --gres=gpu:L40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH -o logs/sss-early-exit-%j.out
#SBATCH -e logs/sss-early-exit-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/ILFC2}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf-cache}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
CUDA_VENV_PATH="${CUDA_VENV_PATH:-$PROJECT_ROOT/.venv-cuda}"
VENV_PATH="${VENV_PATH:-$CUDA_VENV_PATH}"

cd "$PROJECT_ROOT"

mkdir -p logs "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME
export TRANSFORMERS_CACHE
export HF_DATASETS_CACHE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Job ${SLURM_JOB_ID:-unknown} on $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "HF_HOME=$HF_HOME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

if [[ -x "$VENV_PATH/bin/python" ]]; then
  run_cmd=("$VENV_PATH/bin/python" -m ilfc)
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  run_cmd=("$PROJECT_ROOT/.venv/bin/python" -m ilfc)
elif command -v poetry >/dev/null 2>&1; then
  run_cmd=(poetry run python -m ilfc)
else
  echo "ERROR: no Python environment found."
  echo "Expected a CUDA venv at $VENV_PATH, a fallback venv at $PROJECT_ROOT/.venv, or Poetry on PATH."
  exit 1
fi

echo "Run command: ${run_cmd[*]}"
"${run_cmd[@]}" early_exit
