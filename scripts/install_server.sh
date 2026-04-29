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

echo "==> Installing TileLang for FLA Hopper backward kernels"
poetry run python -m pip install \
  --no-cache-dir \
  "tilelang"

echo "==> Installing causal-conv1d against this PyTorch/CUDA"
MAX_JOBS="${MAX_JOBS:-4}" poetry run python -m pip install \
  --no-build-isolation \
  --no-cache-dir \
  "causal-conv1d>=1.6.1"

echo "==> Final import check"
poetry run python - <<'PY'
import importlib.util
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

missing = []

for name in ["fla", "fla.ops", "fla.layers", "causal_conv1d", "tilelang"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {spec.origin if spec else None}")
    if spec is None:
        missing.append(name)

if missing:
    raise SystemExit(f"Missing required fast-path imports: {missing}")

print()
print("Fast-path dependencies look installed.")
PY

echo "==> Version check"
poetry run python - <<'PY'
import torch
import triton
import tilelang

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("triton:", triton.__version__)
print("tilelang:", getattr(tilelang, "__version__", "unknown"))

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "==> Done"