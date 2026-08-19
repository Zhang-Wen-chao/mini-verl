#!/usr/bin/env bash
# Wait for the pinned official verl environment to finish syncing, verify that
# it is usable, then launch the reproducible 4-GPU GRPO smoke exactly once.
#
# This avoids racing uv's atomic environment update on a shared filesystem.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${VERL_DIR:?Set VERL_DIR to the pinned official verl checkout.}"
: "${MODEL_PATH:?Set MODEL_PATH to the complete Qwen3-0.6B snapshot.}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the smoke train parquet.}"
: "${TEST_FILE:?Set TEST_FILE to the smoke validation parquet.}"

VENV_PATH=${VENV_PATH:-"$VERL_DIR/.venv-official-fsdp-vllm"}
RUN_ROOT=${RUN_ROOT:-"$(pwd)/artifacts/qwen3-0.6b-gsm8k-grpo-4gpu-smoke"}
WAIT_TIMEOUT_SECONDS=${WAIT_TIMEOUT_SECONDS:-14400}
LOCK_PATH="$VENV_PATH/.lock"

mkdir -p "$RUN_ROOT/logs"
LOG_PATH="$RUN_ROOT/logs/supervisor.log"
exec >>"$LOG_PATH" 2>&1

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
echo "$(timestamp) supervisor_started venv=$VENV_PATH"

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while ! flock -n "$LOCK_PATH" -c true; do
  if (( SECONDS >= deadline )); then
    echo "$(timestamp) supervisor_timeout waiting_for_uv_lock"
    exit 1
  fi
  echo "$(timestamp) waiting_for_uv_sync"
  sleep 30
done

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  echo "$(timestamp) supervisor_failed missing_venv_python"
  exit 1
fi

"$VENV_PATH/bin/python" - <<'PY'
import importlib
import torch

for module in ("torch", "transformers", "ray", "vllm", "verl"):
    importlib.import_module(module)
if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
    raise RuntimeError(f"Expected four CUDA devices, got {torch.cuda.device_count()}")
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": torch.cuda.device_count()})
PY

echo "$(timestamp) official_runtime_verified launching_grpo"
source "$VENV_PATH/bin/activate"
exec bash "$SCRIPT_DIR/run_qwen3_0_6b_4gpu_smoke.sh"
