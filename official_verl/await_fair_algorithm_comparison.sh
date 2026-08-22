#!/usr/bin/env bash
# One PPO or standard-GRPO development leg under an identical rollout budget.
# It waits for idle GPUs and never alters foreign processes.
set -euo pipefail

: "${ROOT:?Set ROOT to the persistent mini-verl-l20 checkout.}"
: "${ALGORITHM:?Set ALGORITHM to ppo or grpo.}"
LOCAL_ROOT=${LOCAL_ROOT:-/tmp/official-verl-local-fsdp-vllm}
TRAINING_STEPS=${TRAINING_STEPS:-5}
SEED=${SEED:-42}
RUN_ID=${RUN_ID:-"qwen3.5-4b-fair-${ALGORITHM}-${TRAINING_STEPS}step-seed${SEED}-$(date +%Y%m%dT%H%M%S)"}
RUN_ROOT=${RUN_ROOT:-"$ROOT/artifacts/$RUN_ID"}
WAIT_SECONDS=${WAIT_SECONDS:-43200}
PYTHON_BIN="$LOCAL_ROOT/venv/bin/python"
VERL_DIR="$LOCAL_ROOT/verl"
WATCH_LOG="$RUN_ROOT/logs/gate.log"

# Frozen shared sample contract: 3 prompts times 4 completions per step.
TRAIN_BATCH_SIZE=3
ROLLOUT_N=4
AGENT_NUM_WORKERS=12
PPO_MINI_BATCH_SIZE=3
# The pinned FSDP1 Critic has no smaller legal global mini-batch at n=4 and
# DP=3 (the global mini-batch must divide evenly over all three ranks).  Keep
# all 12 trajectories and offload only saved Critic activations to CPU.
PPO_CRITIC_ACTIVATION_OFFLOAD=true
TRAINER_GPUS=3
ROLLOUT_GPUS=1
ROLLOUT_TP=1
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=384
MAX_MODEL_LEN=896
MAX_NUM_BATCHED_TOKENS=896

case "$ALGORITHM" in ppo|grpo) ;; *) echo "ALGORITHM must be ppo or grpo" >&2; exit 2 ;; esac
(( TRAINING_STEPS > 0 )) || { echo "TRAINING_STEPS must be positive" >&2; exit 2; }
mkdir -p "$RUN_ROOT/logs"
log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$WATCH_LOG"; }
stop() { log "GATE_FAILED: $*"; printf '%s\n' "$*" >"$RUN_ROOT/logs/gate_failure"; exit 1; }

[[ -x "$PYTHON_BIN" ]] || stop "missing runtime interpreter"
[[ -f "$VERL_DIR/UPSTREAM_COMMIT" && -f "$VERL_DIR/uv.lock" ]] || stop "missing local pinned runtime"
[[ -f "$ROOT/.official-verl/verl/UPSTREAM_COMMIT" && -f "$ROOT/.official-verl/verl/uv.lock" ]] || stop "missing persistent pinned runtime"
cmp -s "$VERL_DIR/UPSTREAM_COMMIT" "$ROOT/.official-verl/verl/UPSTREAM_COMMIT" || stop "local and persistent revisions differ"
cmp -s "$VERL_DIR/uv.lock" "$ROOT/.official-verl/verl/uv.lock" || stop "local and persistent locks differ"

deadline=$(( $(date +%s) + WAIT_SECONDS ))
log "waiting for four idle GPUs; this watcher will not alter foreign processes"
while true; do
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
    (( $(date +%s) < deadline )) || stop "timed out waiting for idle GPUs"
    sleep 30
  done
  if ! "$PYTHON_BIN" "$ROOT/official_verl/preflight.py" --verl-dir "$VERL_DIR" --require-cuda --require-runtime --require-lock >"$RUN_ROOT/logs/preflight.json"; then
    stop "runtime preflight failed"
  fi
  # CUDA/Ray preflight itself takes long enough for another shared user to claim
  # the GPUs. Re-check immediately before creating any Ray or vLLM process.
  # A busy GPU here returns to the idle wait instead of starting a mixed-tenant
  # run or requiring a human to restart the overnight watcher.
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    log "GPU became busy during runtime preflight; returning to idle wait"
    continue
  fi
  break
done

common=(
  "VERL_DIR=$VERL_DIR" "MODEL_PATH=$ROOT/.official-verl/models/Qwen3.5-4B"
  "TRAIN_FILE=$ROOT/.official-verl/data/qwen3_5_4b/training-2037-processor-filtered-openr1-math-v5-short-20260819T1919.parquet"
  "TEST_FILE=$ROOT/.official-verl/data/qwen3_5_4b/calibration-math-test-64-v3-short.parquet"
  "PYTHON_BIN=$PYTHON_BIN" "COMPAT_PATH=$ROOT/official_verl/compat" "RUN_ROOT=$RUN_ROOT"
  "TRAINING_STEPS=$TRAINING_STEPS" "SAVE_FREQ=$TRAINING_STEPS" "TEST_FREQ=$TRAINING_STEPS"
  "TRAINER_GPUS=$TRAINER_GPUS" "ROLLOUT_GPUS=$ROLLOUT_GPUS" "ROLLOUT_TP=$ROLLOUT_TP"
  "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE" "PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE"
  "ROLLOUT_N=$ROLLOUT_N" "AGENT_NUM_WORKERS=$AGENT_NUM_WORKERS"
  "MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH" "MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
  "MAX_MODEL_LEN=$MAX_MODEL_LEN" "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
  "MINI_VERL_FORCE_MMAP_WEIGHT_TRANSFER=1" "MINI_VERL_WEIGHT_TRANSFER_MMAP_DIR=/tmp"
  "NCCL_SHM_DISABLE=1" "CUDA_DEVICE_MAX_CONNECTIONS=1"
)
seed_overrides=("actor_rollout_ref.actor.fsdp_config.seed=$SEED" "actor_rollout_ref.actor.data_loader_seed=$SEED" "actor_rollout_ref.rollout.seed=$SEED" "trainer.val_before_train=True")

set +e
if [[ "$ALGORITHM" == ppo ]]; then
  log "starting PPO with 12 trajectories per step, real GAE Critic, and Critic activation offload"
  env "${common[@]}" "CRITIC_MODEL_PATH=$ROOT/.official-verl/models/Qwen3.5-4B" "CRITIC_ACTIVATION_OFFLOAD=$PPO_CRITIC_ACTIVATION_OFFLOAD" bash "$ROOT/official_verl/run_qwen3_5_4b_ppo_gae_calibration.sh" "${seed_overrides[@]}" "critic.data_loader_seed=$SEED"
else
  log "starting standard GRPO with the same 12 trajectories per step"
  env "${common[@]}" "PROJECT_NAME=official-verl-grpo-comparison" "EXPERIMENT_NAME=qwen3.5-4b-fair-standard-grpo" bash "$ROOT/official_verl/run_qwen3_5_4b_4gpu_calibration.sh" "${seed_overrides[@]}" "algorithm.norm_adv_by_std_in_grpo=true"
fi
launch_status=$?
set -e
printf '%s\n' "$launch_status" >"$RUN_ROOT/logs/watch_exit_status"
(( launch_status == 0 )) || stop "$ALGORITHM launcher exited with status $launch_status"

if ! "$PYTHON_BIN" - "$RUN_ROOT" "$ALGORITHM" "$TRAINING_STEPS" <<'PY'; then
import json, math, re, sys
from pathlib import Path

run_root, algorithm, steps = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
text = (run_root / "logs/train.log").read_text(encoding="utf-8", errors="replace")
text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
expected = "gae" if algorithm == "ppo" else "grpo"
if f"'adv_estimator': '{expected}'" not in text:
    raise SystemExit("resolved advantage estimator differs from request")
if algorithm == "ppo" and "Disabled critic as algorithm.adv_estimator != gae" in text:
    raise SystemExit("PPO Critic was disabled")
if re.search(r"(?:CUDA out of memory|OutOfMemoryError|\bOOM\b|\bnan\b|\binf\b)", text, re.I):
    raise SystemExit("log contains OOM or non-finite marker")
lines = [line for line in text.splitlines() if "training/global_step:" in line]
seen = {int(m.group(1)) for line in lines if (m := re.search(r"training/global_step:(\d+)", line))}
missing_steps = sorted(set(range(1, steps + 1)) - seen)
if missing_steps:
    raise SystemExit(f"missing training metrics for steps {missing_steps}")
final = [line for line in lines if f"training/global_step:{steps}" in line][-1]
required = ["actor/loss:", "actor/ppo_kl:"]
if algorithm == "ppo": required += ["critic/vf_loss:", "critic/values/mean:", "critic/advantages/mean:"]
missing = [name for name in required if name not in final]
if missing: raise SystemExit(f"incomplete final update: {missing}")
number = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
summary = {"algorithm": algorithm, "steps": steps, "rollouts_per_step": 12}
for name in ("actor/loss", "actor/ppo_kl", "critic/vf_loss", "critic/values/mean", "critic/advantages/mean", "critic/rewards/mean", "perf/total_num_tokens", "timing_s/step", "actor/perf/max_memory_allocated_gb", "critic/perf/max_memory_allocated_gb"):
    if match := re.search(re.escape(name) + r":(?:np\.float64\()?" + number, final):
        value = float(match.group(1))
        if not math.isfinite(value): raise SystemExit(f"non-finite {name}")
        summary[name] = value
if not (run_root / "checkpoints" / f"global_step_{steps}").is_dir(): raise SystemExit("final checkpoint is missing")
if not any((run_root / "rollout_samples").glob("*.jsonl")): raise SystemExit("rollout sample is missing")
(run_root / "logs/comparison_health_summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
  stop "$ALGORITHM artifact or numerical-health gate failed"
fi
log "FAIR_COMPARISON_LEG_FINISHED: $ALGORITHM completed $TRAINING_STEPS finite updates under the shared 12-trajectory contract"
