#!/usr/bin/env bash
# Wait for all four L20 GPUs to be empty, then run exactly one PPO/GAE health
# gate. It never kills, cleans up, or otherwise alters foreign processes.
set -euo pipefail

: "${ROOT:?Set ROOT to the persistent mini-verl-l20 checkout.}"
LOCAL_ROOT=${LOCAL_ROOT:-/tmp/official-verl-local-fsdp-vllm}
RUN_ID=${RUN_ID:-"qwen3.5-4b-openr1-ppo-gae-1step-trainer3-rollout1-$(date +%Y%m%dT%H%M%S)"}
RUN_ROOT=${RUN_ROOT:-"$ROOT/artifacts/$RUN_ID"}
WAIT_SECONDS=${WAIT_SECONDS:-43200}
PYTHON_BIN="$LOCAL_ROOT/venv/bin/python"
VERL_DIR="$LOCAL_ROOT/verl"
WATCH_LOG="$RUN_ROOT/logs/gate.log"

mkdir -p "$RUN_ROOT/logs"
log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$WATCH_LOG"; }
stop() { log "GATE_FAILED: $*"; printf '%s\n' "$*" >"$RUN_ROOT/logs/gate_failure"; exit 1; }

[[ -x "$PYTHON_BIN" ]] || stop "missing runtime interpreter: $PYTHON_BIN"
[[ -f "$LOCAL_ROOT/status" && "$(<"$LOCAL_ROOT/status")" == local_sync_complete ]] || stop "missing successful local official-verl bootstrap"

deadline=$(( $(date +%s) + WAIT_SECONDS ))
log "waiting for four idle GPUs; this watcher will not alter foreign processes"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
  (( $(date +%s) < deadline )) || stop "timed out waiting for idle GPUs"
  sleep 30
done

log "GPUs are idle; running locked CUDA/runtime preflight"
if ! "$PYTHON_BIN" "$ROOT/official_verl/preflight.py" --verl-dir "$VERL_DIR" --require-cuda --require-runtime --require-lock >"$RUN_ROOT/logs/preflight.json"; then
  stop "runtime preflight failed"
fi

log "starting one-step 4B PPO/GAE actor-critic feasibility calibration"
set +e
env VERL_DIR="$VERL_DIR" MODEL_PATH="$ROOT/.official-verl/models/Qwen3.5-4B" \
  CRITIC_MODEL_PATH="$ROOT/.official-verl/models/Qwen3.5-4B" \
  TRAIN_FILE="$ROOT/.official-verl/data/qwen3_5_4b/training-2037-processor-filtered-openr1-math-v5-short-20260819T1919.parquet" \
  TEST_FILE="$ROOT/.official-verl/data/qwen3_5_4b/calibration-math-test-64-v3-short.parquet" \
  PYTHON_BIN="$PYTHON_BIN" COMPAT_PATH="$ROOT/official_verl/compat" RUN_ROOT="$RUN_ROOT" \
  TRAINING_STEPS=1 SAVE_FREQ=1 TEST_FREQ=1 TRAINER_GPUS=3 ROLLOUT_GPUS=1 ROLLOUT_TP=1 \
  TRAIN_BATCH_SIZE=3 PPO_MINI_BATCH_SIZE=3 ROLLOUT_N=1 AGENT_NUM_WORKERS=3 \
  MAX_PROMPT_LENGTH=512 MAX_RESPONSE_LENGTH=384 MAX_MODEL_LEN=896 MAX_NUM_BATCHED_TOKENS=896 \
  MINI_VERL_FORCE_MMAP_WEIGHT_TRANSFER=1 MINI_VERL_WEIGHT_TRANSFER_MMAP_DIR=/tmp \
  bash "$ROOT/official_verl/run_qwen3_5_4b_ppo_gae_calibration.sh"
launch_status=$?
set -e
printf '%s\n' "$launch_status" >"$RUN_ROOT/logs/watch_exit_status"
(( launch_status == 0 )) || stop "PPO launcher exited with status $launch_status"

if ! "$PYTHON_BIN" - "$RUN_ROOT" <<'PY'; then
import math
import re
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
log = (run_root / "logs/train.log").read_text(encoding="utf-8", errors="replace")
log = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log)
if "'adv_estimator': 'gae'" not in log:
    raise SystemExit("resolved config did not select GAE")
if "Disabled critic as algorithm.adv_estimator != gae" in log:
    raise SystemExit("log says Critic was disabled")
if re.search(r"(?:CUDA out of memory|OutOfMemoryError|\bOOM\b|\bnan\b|\binf\b)", log, re.IGNORECASE):
    raise SystemExit("log contains OOM or non-finite marker")
lines = [line for line in log.splitlines() if "training/global_step:1" in line]
if not lines:
    raise SystemExit("global step 1 metrics are missing")
line = lines[-1]
required = ("critic/vf_loss:", "critic/values/mean:", "critic/advantages/mean:", "actor/loss:", "actor/ppo_kl:")
missing = [metric for metric in required if metric not in line]
if missing:
    raise SystemExit(f"step-1 metrics prove no full Critic update: missing {missing}")
number = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
summary = {}
for metric in ("critic/vf_loss", "critic/values/mean", "critic/advantages/mean", "actor/loss", "actor/ppo_kl", "timing_s/step", "actor/perf/max_memory_allocated_gb", "critic/perf/max_memory_allocated_gb"):
    match = re.search(re.escape(metric) + r":(?:np\.float64\()?" + number, line)
    if match is not None:
        value = float(match.group(1))
        if not math.isfinite(value):
            raise SystemExit(f"non-finite {metric}")
        summary[metric] = value
checkpoint = run_root / "checkpoints/global_step_1"
if not checkpoint.is_dir():
    raise SystemExit("global_step_1 checkpoint is missing")
if not any((run_root / "rollout_samples").glob("*.jsonl")):
    raise SystemExit("rollout sample is missing")
(run_root / "logs/ppo_health_summary.txt").write_text(repr(summary) + "\n", encoding="utf-8")
print(summary)
PY
  stop "PPO artifact or numerical-health gate failed"
fi
log "PPO_GAE_CALIBRATION_FINISHED: real Critic and one finite GAE update verified"
