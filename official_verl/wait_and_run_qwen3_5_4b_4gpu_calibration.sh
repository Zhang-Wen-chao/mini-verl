#!/usr/bin/env bash
# Wait until all four L20 GPUs are clear, then launch one bounded calibration.
# It never kills, cleans up, or otherwise alters foreign processes.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${RUN_ROOT:?Set a new RUN_ROOT for this attempt.}"
: "${VERL_DIR:?Set VERL_DIR to the pinned official verl checkout.}"
: "${MODEL_PATH:?Set MODEL_PATH to the complete Qwen3.5-4B snapshot.}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the calibration parquet.}"
: "${TEST_FILE:?Set TEST_FILE to the validation parquet.}"

CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-15}
WAIT_TIMEOUT_SECONDS=${WAIT_TIMEOUT_SECONDS:-7200}
LOCK_PATH=${LOCK_PATH:-"$RUN_ROOT/.launch.lock"}
mkdir -p "$RUN_ROOT/logs"
LOG_PATH="$RUN_ROOT/logs/watcher.log"

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "[$(timestamp)] another watcher already owns this RUN_ROOT" >&2
  exit 2
fi

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
echo "[$(timestamp)] watcher started; it will not kill or alter foreign processes" >>"$LOG_PATH"
while :; do
  if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    echo "[$(timestamp)] all GPUs empty; starting official GRPO calibration" >>"$LOG_PATH"
    exec bash "$SCRIPT_DIR/run_qwen3_5_4b_4gpu_calibration.sh"
  fi
  if (( SECONDS >= deadline )); then
    echo "[$(timestamp)] timeout: compute process still present" >>"$LOG_PATH"
    exit 1
  fi
  echo "[$(timestamp)] waiting: compute process present" >>"$LOG_PATH"
  sleep "$CHECK_INTERVAL_SECONDS"
done
