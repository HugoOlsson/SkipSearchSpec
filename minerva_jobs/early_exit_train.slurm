#!/bin/bash
#SBATCH -J sss-early-exit
#SBATCH -p long
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH -o logs/sss-early-exit-%j.out
#SBATCH -e logs/sss-early-exit-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/SkipSearchSpec}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf-cache}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"

cd "$PROJECT_ROOT"

mkdir -p logs checkpoints "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME
export TRANSFORMERS_CACHE
export HF_DATASETS_CACHE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"

echo "Job ${SLURM_JOB_ID:-unknown} on $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "HF_HOME=$HF_HOME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  echo "ERROR: expected Poetry environment at $VENV_PATH"
  echo "Create it with:"
  echo "  cd $PROJECT_ROOT"
  echo "  poetry config virtualenvs.in-project true --local"
  echo "  poetry install --sync"
  exit 1
fi

run_cmd=("$VENV_PATH/bin/python" -u -m skip_search_spec.main)
echo "Run command: ${run_cmd[*]}"
"${run_cmd[@]}" train_early_exit
