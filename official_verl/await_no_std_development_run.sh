#!/usr/bin/env bash
# Advance a no-std GRPO experiment from a successful 20-step compatibility
# calibration to one bounded 170-step development run.  This is deliberately
# not an automatic quality claim, seed sweep, or held-out MATH evaluation.
set -euo pipefail

: "${ROOT:?Set ROOT to the persistent mini-verl-l20 checkout.}"
: "${CALIBRATION_ROOT:?Set CALIBRATION_ROOT to the completed 20-step run.}"
LOCAL_ROOT=${LOCAL_ROOT:-/tmp/official-verl-local-fsdp-vllm}
RUN_ID=${RUN_ID:-"qwen3.5-4b-openr1-grpo-nostd-170step-trainer3-rollout1-$(date +%Y%m%dT%H%M%S)"}
RUN_ROOT=${RUN_ROOT:-"$ROOT/artifacts/$RUN_ID"}
WAIT_SECONDS=${WAIT_SECONDS:-43200}

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

calibration_log="$CALIBRATION_ROOT/logs/train.log"
deadline=$(( $(date +%s) + WAIT_SECONDS ))
log "waiting for bounded 20-step calibration: $CALIBRATION_ROOT"
while [[ ! -f "$CALIBRATION_ROOT/logs/watch_exit_status" ]]; do
  [[ -f "$CALIBRATION_ROOT/logs/gate_failure" ]] && stop "calibration gate failed: $(<"$CALIBRATION_ROOT/logs/gate_failure")"
  if (( $(date +%s) >= deadline )); then
    stop "calibration did not finish before the deadline"
  fi
  sleep 30
done

[[ "$(<"$CALIBRATION_ROOT/logs/watch_exit_status")" == "0" ]] || stop "calibration launcher failed"
[[ -f "$CALIBRATION_ROOT/logs/exit_status" ]] || stop "calibration did not write exit_status"
[[ "$(<"$CALIBRATION_ROOT/logs/exit_status")" == "0" ]] || stop "calibration training process failed"
[[ -d "$CALIBRATION_ROOT/checkpoints/global_step_20" ]] || stop "calibration checkpoint global_step_20 is missing"

rollout_count=$(find "$CALIBRATION_ROOT/rollout_samples" -maxdepth 1 -type f -name '*.jsonl' | wc -l | tr -d ' ')
(( rollout_count >= 20 )) || stop "expected at least 20 calibration rollout files, found $rollout_count"
[[ -s "$calibration_log" ]] || stop "calibration train log is missing"

# Parse the emitted per-step metrics, rather than treating exit status as a
# sufficient health signal.  A zero gradient is permitted for all-zero/all-one
# reward groups, but not for a mixed group.
if ! python3 - "$calibration_log" <<'PY'
import math
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
lines = [line for line in text.splitlines() if "training/global_step:" in line]
if not any(re.search(r"training/global_step:20(?:\D|$)", line) for line in lines):
    raise SystemExit("global step 20 metric line is missing")
if re.search(r"(?:CUDA out of memory|OutOfMemoryError|\bOOM\b|\bnan\b|\binf\b)", text, re.IGNORECASE):
    raise SystemExit("log contains OOM or non-finite marker")

number = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
mixed = 0
mixed_nonzero_grad = 0
clip_values = []
for line in lines:
    score = re.search(r"critic/score/mean:(?:np\.float64\()?" + number, line)
    grad = re.search(r"actor/grad_norm:(?:np\.float64\()?" + number, line)
    kl = re.search(r"actor/ppo_kl:(?:np\.float64\()?" + number, line)
    clip = re.search(r"response_length/clip_ratio:" + number, line)
    if kl is not None and not math.isfinite(float(kl.group(1))):
        raise SystemExit("non-finite PPO KL")
    if clip is not None:
        clip_values.append(float(clip.group(1)))
    if score is not None and grad is not None and 0.0 < float(score.group(1)) < 1.0:
        mixed += 1
        mixed_nonzero_grad += float(grad.group(1)) > 0.0
if mixed and not mixed_nonzero_grad:
    raise SystemExit("mixed-reward calibration groups had no non-zero gradient")
if len(clip_values) >= 10 and all(value >= 0.75 for value in clip_values[-10:]):
    raise SystemExit("last ten calibration steps were severely response-capped")
print({"metric_steps": len(lines), "mixed_groups": mixed, "mixed_nonzero_grad": mixed_nonzero_grad, "last_clip_ratio": clip_values[-1] if clip_values else None})
PY
then
  stop "calibration metrics failed the numerical-health gate"
fi

[[ -x "$PYTHON_BIN" ]] || stop "missing runtime interpreter: $PYTHON_BIN"
log "calibration passed artifact and numerical-health gates; running preflight"
if ! "$PYTHON_BIN" "$ROOT/official_verl/preflight.py" \
  --verl-dir "$VERL_DIR" --require-cuda --require-runtime --require-lock \
  >"$PREFLIGHT_REPORT"; then
  stop "preflight failed; inspect $PREFLIGHT_REPORT"
fi

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  stop "refusing to launch because a GPU has a compute process"
fi

log "all gates passed; starting one 170-step no-std GRPO development run at $RUN_ROOT"
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
  EXPERIMENT_NAME=qwen3.5-4b-openr1-nostd-170step \
  TRAINING_STEPS=170 SAVE_FREQ=170 TEST_FREQ=170 \
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
  stop "development launcher exited with status $launch_status"
fi
log "DEVELOPMENT_RUN_FINISHED: launcher exited successfully; no held-out evaluation was run"
