#!/usr/bin/env bash
# Gate and launch exactly one no-std GRPO compatibility calibration after the
# disposable local official-verl runtime has been rebuilt.  Designed to be run
# under nohup on the L20 host: it never starts a longer quality experiment.
set -euo pipefail

: "${ROOT:?Set ROOT to the persistent mini-verl-l20 checkout.}"
LOCAL_ROOT=${LOCAL_ROOT:-/tmp/official-verl-local-fsdp-vllm}
RUN_ID=${RUN_ID:-"qwen3.5-4b-openr1-grpo-nostd-20step-trainer3-rollout1-$(date +%Y%m%dT%H%M%S)"}
RUN_ROOT=${RUN_ROOT:-"$ROOT/artifacts/$RUN_ID"}
WAIT_SECONDS=${WAIT_SECONDS:-28800}

PYTHON_BIN="$LOCAL_ROOT/venv/bin/python"
VERL_DIR="$LOCAL_ROOT/verl"
STATUS_FILE="$LOCAL_ROOT/status"
WATCH_LOG="$RUN_ROOT/logs/gate.log"
PREFLIGHT_REPORT="$RUN_ROOT/logs/preflight.json"

mkdir -p "$RUN_ROOT/logs"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$WATCH_LOG"
}

stop() {
  log "GATE_FAILED: $*"
  printf '%s\n' "$*" >"$RUN_ROOT/logs/gate_failure"
  exit 1
}

deadline=$(( $(date +%s) + WAIT_SECONDS ))
log "waiting for bootstrap status at $STATUS_FILE (deadline ${WAIT_SECONDS}s)"
while [[ ! -f "$STATUS_FILE" ]]; do
  if (( $(date +%s) >= deadline )); then
    stop "bootstrap did not complete before the deadline"
  fi
  # If the bootstrap itself vanished before it could declare success, waiting
  # would only hide the first actionable failure.
  if ! pgrep -f 'bootstrap_local_official_env.sh|uv sync --python 3.12' >/dev/null; then
    stop "bootstrap process exited without writing status"
  fi
  sleep 30
done

[[ "$(<"$STATUS_FILE")" == "local_sync_complete" ]] || stop "unexpected bootstrap status: $(<"$STATUS_FILE")"
[[ -x "$PYTHON_BIN" ]] || stop "missing runtime interpreter: $PYTHON_BIN"

log "bootstrap declared success; checking imports, pin, and CUDA without loading a model"
if ! "$PYTHON_BIN" "$ROOT/official_verl/preflight.py" \
  --verl-dir "$VERL_DIR" --require-cuda --require-runtime --require-lock \
  >"$PREFLIGHT_REPORT"; then
  stop "preflight failed; inspect $PREFLIGHT_REPORT"
fi

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  stop "refusing to launch because a GPU has a compute process"
fi

log "all gates passed; starting exactly one 20-step no-std GRPO calibration at $RUN_ROOT"
set +e
env \
  VERL_DIR="$VERL_DIR" \
  MODEL_PATH="$ROOT/.official-verl/models/Qwen3.5-4B" \
  TRAIN_FILE="$ROOT/.official-verl/data/qwen3_5_4b/training-2037-processor-filtered-openr1-math-v5-short-20260819T1919.parquet" \
  TEST_FILE="$ROOT/.official-verl/data/qwen3_5_4b/calibration-math-test-64-v3-short.parquet" \
  PYTHON_BIN="$PYTHON_BIN" \
  COMPAT_PATH="$ROOT/official_verl/compat" \
  RUN_ROOT="$RUN_ROOT" \
  PROJECT_NAME=official-verl-grpo \
  EXPERIMENT_NAME=qwen3.5-4b-openr1-nostd-20step \
  TRAINING_STEPS=20 SAVE_FREQ=20 TEST_FREQ=20 \
  TRAINER_GPUS=3 ROLLOUT_GPUS=1 ROLLOUT_TP=1 \
  TRAIN_BATCH_SIZE=3 PPO_MINI_BATCH_SIZE=3 ROLLOUT_N=4 AGENT_NUM_WORKERS=12 \
  MAX_PROMPT_LENGTH=512 MAX_RESPONSE_LENGTH=384 MAX_MODEL_LEN=896 MAX_NUM_BATCHED_TOKENS=896 \
  ACTOR_PARAM_OFFLOAD=false \
  MINI_VERL_FORCE_MMAP_WEIGHT_TRANSFER=1 \
  MINI_VERL_WEIGHT_TRANSFER_MMAP_DIR=/tmp \
  bash "$ROOT/official_verl/run_qwen3_5_4b_no_std_calibration.sh"
launch_status=$?
set -e

printf '%s\n' "$launch_status" >"$RUN_ROOT/logs/watch_exit_status"
if (( launch_status != 0 )); then
  stop "calibration launcher exited with status $launch_status"
fi
log "CALIBRATION_FINISHED: launcher exited successfully; no longer experiment was started"
