# skip_search_spec

Poetry-managed Python project.

## RunPod startup

Use `scripts/runpod_startup.sh` as the pod startup script. It installs the
ephemeral system packages each time, then keeps the repo, Poetry install,
virtualenv, Hugging Face caches, Torch caches, Triton caches, and generated
project outputs on the persistent `/workspace` volume.

For a fresh RunPod template, the startup command can be:

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/HugoOlsson/SkipSearchSpec/main/scripts/runpod_startup.sh)"
```

Useful options:

```bash
HF_TOKEN=hf_... RUNPOD_UPDATE_REPO=1 bash -c "$(curl -fsSL https://raw.githubusercontent.com/HugoOlsson/SkipSearchSpec/main/scripts/runpod_startup.sh)"
```

`RUNPOD_UPDATE_REPO=1` fetches and ff-only pulls if the repo has no local
changes, `RUNPOD_RUN_LOCK=0` skips `poetry lock`, and `RUNPOD_FORCE_INSTALL=1`
reruns `scripts/install_server.sh` even if `.venv` looks ready.
