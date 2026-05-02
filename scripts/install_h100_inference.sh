#!/usr/bin/env bash
set -euo pipefail

# H100 inference-oriented setup.
#
# This intentionally differs from install_server.sh:
# - Uses the older Qwen3 benchmark-style stack for batch-1 decoding.
# - Installs Torch before CUDA attention extensions.
# - Installs flash-attn, which normal Qwen attention can actually use.
#
# Important: pyproject.toml currently allows torch>=2.10. Poetry may try to
# restore that on later `poetry install` runs. Re-run this script after any
# Poetry dependency refresh if you want to keep this inference stack.

echo "==> Installing normal project deps without server extras"
poetry install --with build

echo "==> Installing build helpers"
poetry run python -m pip install "setuptools<82" wheel packaging ninja

echo "==> Installing H100 inference Torch stack"
poetry run python -m pip install \
  --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0"

echo "==> Installing Qwen/Transformers + FlashAttention stack"
poetry run python -m pip install \
  --force-reinstall \
  "transformers==4.51.3" \
  "accelerate>=1.6.0,<2.0.0"

MAX_JOBS="${MAX_JOBS:-4}" poetry run python -m pip install \
  --no-build-isolation \
  --no-cache-dir \
  "flash-attn==2.7.4.post1"

echo "==> Verifying CUDA + attention stack"
poetry run python - <<'PY'
import importlib.util
import torch
import transformers

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)

if not torch.cuda.is_available():
    raise SystemExit("Expected CUDA to be available, but torch.cuda.is_available() is False.")

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

for name in ["flash_attn", "flash_attn_2_cuda"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}:", spec.origin if spec else None)
    if spec is None:
        raise SystemExit(f"Missing required import: {name}")
PY

echo "==> Running tiny FlashAttention model-load smoke test"
poetry run python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).cuda().eval()

print("param device:", next(model.parameters()).device)
print("param dtype:", next(model.parameters()).dtype)
print("attn implementation:", getattr(model.config, "_attn_implementation", None))
print("vocab size:", len(tok))
PY

echo "==> Done"
