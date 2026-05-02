#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing normal project deps + server extras"
poetry install --with build -E server

echo "==> Installing build helpers"
poetry run python -m pip install "setuptools<82" wheel packaging ninja

echo "==> Checking system CUDA compiler"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc was not found."
  echo "Use a CUDA devel image/container, not only a runtime image."
  exit 1
fi

nvcc -V

echo "==> Forcing PyTorch CUDA 12.8 wheel to match system nvcc 12.8"
poetry run python -m pip install \
  --force-reinstall \
  --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0"

echo "==> Verifying PyTorch CUDA version"
poetry run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.version.cuda != "12.8":
    raise SystemExit(
        f"Expected torch.version.cuda == '12.8', got {torch.version.cuda!r}"
    )

if not torch.cuda.is_available():
    raise SystemExit("Expected CUDA to be available, but torch.cuda.is_available() is False.")
PY

echo "==> Core import check"
poetry run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

print()
print("Core CUDA/PyTorch dependencies look installed.")
PY

echo "==> Version check"
poetry run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)

try:
    import triton
    print("triton:", triton.__version__)
except ImportError:
    print("triton: not installed")

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "==> Done"