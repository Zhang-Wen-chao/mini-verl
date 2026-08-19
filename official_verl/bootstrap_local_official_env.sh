#!/usr/bin/env bash
# Materialize the pinned official-verl FSDP + vLLM environment on the container
# local disk.  The project checkout, model, data, and run artifacts remain on
# persistent storage; only the reproducible Python environment and uv cache are
# local and can be rebuilt from the pinned source + lock.
set -euo pipefail

: "${VERL_SOURCE_DIR:?Set VERL_SOURCE_DIR to the pinned official verl checkout.}"
: "${VENDOR_DIR:?Set VENDOR_DIR to the pinned mbridge/TransferQueue sources.}"

LOCAL_ROOT=${LOCAL_ROOT:-/tmp/official-verl-local-fsdp-vllm}
LOCAL_VERL_DIR=${LOCAL_VERL_DIR:-"$LOCAL_ROOT/verl"}
LOCAL_VENV=${LOCAL_VENV:-"$LOCAL_ROOT/venv"}
LOCAL_UV_CACHE=${LOCAL_UV_CACHE:-"$LOCAL_ROOT/uv-cache"}
LOG_PATH=${LOG_PATH:-"$LOCAL_ROOT/logs/uv-sync.log"}

mkdir -p "$LOCAL_ROOT/logs"

# Keep an existing, complete local source copy if it is the same pinned commit.
# A partial prior bootstrap is harmless: uv sync will make the venv consistent.
if [[ ! -f "$LOCAL_VERL_DIR/UPSTREAM_COMMIT" ]] || ! cmp -s "$VERL_SOURCE_DIR/UPSTREAM_COMMIT" "$LOCAL_VERL_DIR/UPSTREAM_COMMIT"; then
  rm -rf "$LOCAL_VERL_DIR"
  cp -a "$VERL_SOURCE_DIR" "$LOCAL_VERL_DIR"
fi

if [[ ! -d "$LOCAL_UV_CACHE" ]]; then
  cp -a "${UV_CACHE_SEED:?Set UV_CACHE_SEED to the warmed official uv cache}" "$LOCAL_UV_CACHE"
fi

cd "$LOCAL_VERL_DIR"
export UV_PROJECT_ENVIRONMENT="$LOCAL_VENV"
export UV_CACHE_DIR="$LOCAL_UV_CACHE"
# Both Git sources are copied at fixed commits into VENDOR_DIR because this L20
# cannot reliably use GitHub's Git transport.  Leave them out of uv's mutation
# set, then install exactly those sources after the lock-resolved sync.
export VERL_UV_NO_INSTALL='mbridge TransferQueue'

uv sync --python 3.12 --extra fsdp --extra vllm --inexact \
  --no-install-package mbridge --no-install-package TransferQueue \
  >>"$LOG_PATH" 2>&1
uv pip install --python "$LOCAL_VENV/bin/python" --no-deps \
  "$VENDOR_DIR/mbridge" "$VENDOR_DIR/TransferQueue" \
  >>"$LOG_PATH" 2>&1

"$LOCAL_VENV/bin/python" - <<'PY' >>"$LOG_PATH" 2>&1
import importlib
import torch

# This experiment uses the FSDP2 + vLLM path, not the Megatron backend.
# `mbridge` is installed at its pinned upstream revision to preserve the source
# lock, but that optional package imports Megatron-Core at module import time.
# Requiring that import here would incorrectly turn a usable FSDP environment
# into a failed bootstrap.
for module in ("torch", "transformers", "ray", "vllm", "verl"):
    importlib.import_module(module)
if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
    raise RuntimeError(f"Expected at least four CUDA GPUs, got {torch.cuda.device_count()}")
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": torch.cuda.device_count()})
PY

printf 'local_sync_complete\n' >"$LOCAL_ROOT/status"
