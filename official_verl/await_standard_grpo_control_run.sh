#!/usr/bin/env bash
# Run one standard-GRPO 170-step control only after the paired no-std
# development run is demonstrably complete and numerically healthy.  Together
# they are a single-seed development comparison, never a held-out result.
set -euo pipefail

: "${ROOT:?Set ROOT to the persistent mini-verl-l20 checkout.}"
: "${NO_STD_ROOT:?Set NO_STD_ROOT to the completed 170-step no-std run.}"
LOCAL_ROOT=${LOCAL_ROOT:-/tmp/official-verl-local-fsdp-vllm}
RUN_ID=${RUN_ID:-"qwen3.5-4b-openr1-grpo-standard-170step-trainer3-rollout1-$(date +%Y%m%dT%H%M%S)"}
RUN_ROOT=${RUN_ROOT:-"$ROOT/artifacts/$RUN_ID"}
WAIT_SECONDS=${WAIT_SECONDS:-57600}

PYTHON_BIN="$LOCAL_ROOT/venv/bin/python"
VERL_DIR="$LOCAL_ROOT/verl"
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

health_gate() {
  local log_path=$1 expected_step=$2
  python3 - "$log_path" "$expected_step" <<'PY'
import math
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
expected_step = int(sys.argv[2])
text = log_path.read_text(encoding="utf-8", errors="replace")
text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
lines = [line for line in text.splitlines() if "training/global_step:" in line]
if not any(re.search(rf"training/global_step:{expected_step}(?:\D|$)", line) for line in lines):
    raise SystemExit(f"global step {expected_step} metric line is missing")
if re.search(r"(?:CUDA out of memory|OutOfMemoryError|\bOOM\b|\bnan\b|\binf\b)", text, re.IGNORECASE):
    raise SystemExit("log contains OOM or non-finite marker")

number = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
finite_kl = 0
mixed = 0
mixed_nonzero_grad = 0
for line in lines:
    score = re.search(r"critic/score/mean:(?:np\.float64\()?" + number, line)
    grad = re.search(r"actor/grad_norm:(?:np\.float64\()?" + number, line)
    kl = re.search(r"actor/ppo_kl:(?:np\.float64\()?" + number, line)
    if kl is not None:
        finite_kl += 1
        if not math.isfinite(float(kl.group(1))):
            raise SystemExit("non-finite PPO KL")
    if score is not None and grad is not None and 0.0 < float(score.group(1)) < 1.0:
        mixed += 1
        mixed_nonzero_grad += float(grad.group(1)) > 0.0
if not finite_kl:
    raise SystemExit("PPO KL was not logged")
if mixed and not mixed_nonzero_grad:
    raise SystemExit("mixed-reward groups had no non-zero gradient")
print({"metric_steps": len(lines), "mixed_groups": mixed, "mixed_nonzero_grad": mixed_nonzero_grad})
PY
}

deadline=$(( $(date +%s) + WAIT_SECONDS ))
log "waiting for no-std development run: $NO_STD_ROOT"
while [[ ! -f "$NO_STD_ROOT/logs/watch_exit_status" ]]; do
  [[ -f "$NO_STD_ROOT/logs/gate_failure" ]] && stop "no-std gate failed: $(<"$NO_STD_ROOT/logs/gate_failure")"
  (( $(date +%s) < deadline )) || stop "no-std development run did not finish before the deadline"
  sleep 30
done

[[ "$(<"$NO_STD_ROOT/logs/watch_exit_status")" == "0" ]] || stop "no-std launcher failed"
[[ -f "$NO_STD_ROOT/logs/exit_status" ]] || stop "no-std run did not write exit_status"
[[ "$(<"$NO_STD_ROOT/logs/exit_status")" == "0" ]] || stop "no-std training process failed"
[[ -d "$NO_STD_ROOT/checkpoints/global_step_170" ]] || stop "no-std global_step_170 checkpoint is missing"
rollout_count=$(find "$NO_STD_ROOT/rollout_samples" -maxdepth 1 -type f -name '*.jsonl' | wc -l | tr -d ' ')
(( rollout_count >= 170 )) || stop "expected at least 170 no-std rollout files, found $rollout_count"
[[ -s "$NO_STD_ROOT/logs/train.log" ]] || stop "no-std train log is missing"
if ! health_gate "$NO_STD_ROOT/logs/train.log" 170; then
  stop "no-std run failed the numerical-health gate"
fi

[[ -x "$PYTHON_BIN" ]] || stop "missing runtime interpreter: $PYTHON_BIN"
log "no-std artifacts and numerical health passed; running standard-control preflight"
if ! "$PYTHON_BIN" "$ROOT/official_verl/preflight.py" \
  --verl-dir "$VERL_DIR" --require-cuda --require-runtime --require-lock \
  >"$PREFLIGHT_REPORT"; then
  stop "preflight failed; inspect $PREFLIGHT_REPORT"
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  stop "refusing to launch because a GPU has a compute process"
fi

log "all gates passed; starting one paired standard-GRPO 170-step control at $RUN_ROOT"
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
  EXPERIMENT_NAME=qwen3.5-4b-openr1-standard-170step-control \
  TRAINING_STEPS=170 SAVE_FREQ=170 TEST_FREQ=170 \
  TRAINER_GPUS=3 ROLLOUT_GPUS=1 ROLLOUT_TP=1 \
  TRAIN_BATCH_SIZE=3 PPO_MINI_BATCH_SIZE=3 ROLLOUT_N=4 AGENT_NUM_WORKERS=12 \
  MAX_PROMPT_LENGTH=512 MAX_RESPONSE_LENGTH=384 MAX_MODEL_LEN=896 MAX_NUM_BATCHED_TOKENS=896 \
  ACTOR_PARAM_OFFLOAD=false \
  MINI_VERL_FORCE_MMAP_WEIGHT_TRANSFER=1 \
  MINI_VERL_WEIGHT_TRANSFER_MMAP_DIR=/tmp \
  bash "$ROOT/official_verl/run_qwen3_5_4b_4gpu_calibration.sh" \
  actor_rollout_ref.actor.fsdp_config.seed=42 \
  actor_rollout_ref.actor.data_loader_seed=42 \
  actor_rollout_ref.rollout.seed=42
launch_status=$?
set -e

printf '%s\n' "$launch_status" >"$RUN_ROOT/logs/watch_exit_status"
if (( launch_status != 0 )); then
  stop "standard-control launcher exited with status $launch_status"
fi
log "STANDARD_CONTROL_FINISHED: paired one-seed development run completed; no held-out evaluation was run"
