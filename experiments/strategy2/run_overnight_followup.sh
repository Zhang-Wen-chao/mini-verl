#!/usr/bin/env bash
# Continue the Strategy 2 study without overwriting a completed experiment.
#
# This runs on the L20 host (not inside the container).  It deliberately
# waits for the active four-arm protocol-fix evaluation to be complete before
# allocating the Ray training cluster.  Once it is safe to proceed it:
#   1. evaluates the four existing arms at 16K context on GPU 0 (capacity
#      ablation, sequential so it uses only one inference GPU);
#   2. trains quality_process_reward_v3 on the repaired answer protocol using
#      GPUs 1--3;
#   3. evaluates v3 at the same 8K protocol as the formal four-arm rerun; and
#   4. writes a compact, machine-readable comparison report.
#
# Source checkpoints, records, metrics, and logs are never removed.  The only
# cleanup is the HF-format `models/` scratch directory created inside the two
# new evaluation roots, and it happens only after a full record/summary check.
set -euo pipefail

BASE=/mnt/storage01/zhangwenchao02
CONTAINER=slime-dev
SCRIPT_DIR="$BASE/experiments/strategy2"
SOURCE_ROOT=${SOURCE_ROOT:-"$BASE/strategy2-eval-protocolfix-20260826T002924"}
RUN_ID=${RUN_ID:-"$(date +%Y%m%dT%H%M%S)"}
NIGHTLY_ROOT=${NIGHTLY_ROOT:-"$BASE/strategy2-overnight-followup-$RUN_ID"}
CONTEXT_ROOT="$NIGHTLY_ROOT/context16k"
V3_EVAL_ROOT="$NIGHTLY_ROOT/v3-8k"
ARM_V3=quality_process_reward_v3
LOG_DIR="$NIGHTLY_ROOT/logs"
REPORT="$NIGHTLY_ROOT/comparison.json"
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-50400} # 14 hours
POLL_SECONDS=${POLL_SECONDS:-120}
MIN_FREE_GIB=${MIN_FREE_GIB:-200}

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/orchestrator.log") 2>&1

timestamp() { date -Is; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
fail() { log "ERROR: $*"; printf 'failed_at=%s\nreason=%s\n' "$(timestamp)" "$*" > "$NIGHTLY_ROOT/FAILED"; exit 1; }

require_free_space() {
  local avail_gib
  avail_gib=$(df -BG --output=avail /mnt/storage01 | awk 'NR==2 {gsub(/G/, "", $1); print $1}')
  [[ "$avail_gib" =~ ^[0-9]+$ ]] || fail "could not determine free disk space"
  (( avail_gib >= MIN_FREE_GIB )) || fail "only ${avail_gib}GiB free; need at least ${MIN_FREE_GIB}GiB"
  log "disk check: ${avail_gib}GiB available"
}

valid_eval_root() {
  local root=$1
  shift
  local arm records
  for arm in "$@"; do
    records="$root/results/$arm/records.jsonl"
    [[ -s "$records" ]] || return 1
    [[ -s "$root/results/$arm/summary.json" ]] || return 1
    [[ "$(wc -l < "$records")" -eq 30 ]] || return 1
  done
}

safe_cleanup_models() {
  local root=$1
  shift
  local models="$root/models"
  valid_eval_root "$root" "$@" || fail "refusing to clean incomplete evaluation root: $root"
  case "$models" in
    "$BASE"/strategy2-overnight-followup-*/context16k/models|"$BASE"/strategy2-overnight-followup-*/v3-8k/models) ;;
    *) fail "refusing to clean unexpected path: $models" ;;
  esac
  if [[ -d "$models" ]]; then
    local size
    size=$(du -sh "$models" | awk '{print $1}')
    rm -rf -- "$models"
    log "removed verified temporary HF models at $models (logical size $size)"
  fi
}

wait_for_formal_eval() {
  local elapsed=0
  local arms=(base outcome_reward process_reward quality_process_reward_v2)
  log "waiting for formal protocol-fix evaluation at $SOURCE_ROOT"
  while (( elapsed <= MAX_WAIT_SECONDS )); do
    [[ ! -e "$SOURCE_ROOT/INCOMPLETE_RESULTS" ]] || fail "the formal evaluation ended incomplete"
    if [[ -s "$SOURCE_ROOT/COMPLETED" ]] && valid_eval_root "$SOURCE_ROOT" "${arms[@]}"; then
      log "formal evaluation is complete and all four result sets validate"
      return
    fi
    local counts=()
    local arm file count
    for arm in "${arms[@]}"; do
      file="$SOURCE_ROOT/results/$arm/records.jsonl"
      count=0
      [[ -f "$file" ]] && count=$(wc -l < "$file")
      counts+=("$arm=$count")
    done
    log "formal evaluation still running: ${counts[*]}"
    sleep "$POLL_SECONDS"
    elapsed=$((elapsed + POLL_SECONDS))
  done
  fail "timed out waiting for formal evaluation after ${MAX_WAIT_SECONDS}s"
}

start_v3_training() {
  local output job_id
  [[ ! -e "$BASE/retool-rl-smoke/$ARM_V3" ]] || fail "v3 output already exists; refusing to overwrite it"
  require_free_space
  log "submitting $ARM_V3 training job on the three-GPU Ray cluster"
  output=$(nerdctl exec -w / "$CONTAINER" bash -lc "
    cd '$SCRIPT_DIR' &&
    REUSE_EXISTING_RAY=1 \\
    RAY_SUBMIT_NO_WAIT=1 \\
    RETOOL_SYNC_CKPT_STAGING=1 \\
    QUALITY_PROCESS_REWARD=1 \\
    RUN_SUFFIX=_v3 \\
    bash reward_compare.sh
  " 2>&1) || { printf '%s\n' "$output"; fail "v3 job submission failed"; }
  printf '%s\n' "$output" | tee "$LOG_DIR/v3-submit.log"
  job_id=$(printf '%s\n' "$output" | sed -nE 's/.*Job submission id: ([[:alnum:]_-]+).*/\1/p' | tail -1)
  [[ -n "$job_id" ]] || fail "could not parse Ray job id from v3 submission"
  printf '%s\n' "$job_id" > "$NIGHTLY_ROOT/v3-ray-job-id"
  log "v3 Ray job id: $job_id"
}

monitor_v3_training() {
  local job_id=$1 status elapsed=0
  local ckpt="$BASE/retool-rl-smoke/$ARM_V3/ckpt/iter_0000019"
  log "monitoring v3 training job $job_id"
  while (( elapsed <= MAX_WAIT_SECONDS )); do
    status=$(nerdctl exec -w / "$CONTAINER" ray job status --address=http://127.0.0.1:8265 "$job_id" 2>&1 || true)
    printf '[%s] ray_job_status=%s\n' "$(timestamp)" "$status" | tee -a "$LOG_DIR/v3-status.log"
    case "$status" in
      *SUCCEEDED*)
        [[ -s "$ckpt/common.pt" ]] || fail "v3 reported success without common.pt"
        [[ -s "$ckpt/.metadata" ]] || fail "v3 reported success without .metadata"
        local shard_count
        shard_count=$(find "$ckpt" -maxdepth 1 -name '*.distcp' -type f -size +1G | wc -l)
        (( shard_count >= 4 )) || fail "v3 checkpoint has only $shard_count large distcp shards"
        log "v3 training completed with a structurally complete checkpoint"
        return
        ;;
      *FAILED*|*STOPPED*)
        fail "v3 Ray job ended with status: $status"
        ;;
    esac
    require_free_space
    sleep "$POLL_SECONDS"
    elapsed=$((elapsed + POLL_SECONDS))
  done
  fail "timed out monitoring v3 training after ${MAX_WAIT_SECONDS}s"
}

run_context16k() {
  local arms=(base outcome_reward process_reward quality_process_reward_v2)
  require_free_space
  log "starting 16K context capacity ablation on GPU 0, sequentially"
  nerdctl exec -w / "$CONTAINER" bash -lc "
    cd '$SCRIPT_DIR' &&
    RUN_ROOT='$CONTEXT_ROOT' \\
    MAX_PROBLEMS=30 \\
    MAX_NEW_TOKENS=1024 \\
    MAX_CONTEXT_TOKENS=16384 \\
    MAX_TURNS=16 \\
    TEMPERATURE=0.0 \\
    CHECKPOINT_ARMS='outcome_reward process_reward quality_process_reward_v2' \\
    bash run_heldout_eval.sh
  " 2>&1 | tee "$LOG_DIR/context16k.log"
  valid_eval_root "$CONTEXT_ROOT" "${arms[@]}" || fail "16K context evaluation did not produce four complete result sets"
  safe_cleanup_models "$CONTEXT_ROOT" "${arms[@]}"
  log "16K context capacity ablation completed"
}

run_v3_eval() {
  local arms=(base "$ARM_V3")
  require_free_space
  log "starting v3 evaluation with the formal 8K protocol on GPU 0"
  nerdctl exec -w / "$CONTAINER" bash -lc "
    cd '$SCRIPT_DIR' &&
    RUN_ROOT='$V3_EVAL_ROOT' \\
    MAX_PROBLEMS=30 \\
    MAX_NEW_TOKENS=1024 \\
    MAX_CONTEXT_TOKENS=8192 \\
    MAX_TURNS=16 \\
    TEMPERATURE=0.0 \\
    CHECKPOINT_ARMS='$ARM_V3' \\
    bash run_heldout_eval.sh
  " 2>&1 | tee "$LOG_DIR/v3-eval.log"
  valid_eval_root "$V3_EVAL_ROOT" "${arms[@]}" || fail "v3 evaluation did not produce complete base and v3 results"
  safe_cleanup_models "$V3_EVAL_ROOT" "${arms[@]}"
  log "v3 8K evaluation completed"
}

write_report() {
  log "writing consolidated comparison report"
  python3 - "$SOURCE_ROOT" "$CONTEXT_ROOT" "$V3_EVAL_ROOT" "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

formal_root, context_root, v3_root, report_path = map(Path, sys.argv[1:])

def summary(root, arm):
    path = root / "results" / arm / "summary.json"
    return json.loads(path.read_text())

def concise(data):
    status = data["terminal_status_counts"]
    total = data["total"]
    return {
        "correct": data["correct"],
        "total": total,
        "accuracy": data["accuracy"],
        "answer_terminal_rate": status.get("answer", 0) / total,
        "context_limit_rate": status.get("context_limit", 0) / total,
        "turn_limit_rate": status.get("turn_limit", 0) / total,
        "tool_use_rate": data["tool_use_rate"],
        "mean_tool_calls": data["mean_tool_calls"],
        "mean_invalid_actions": data["mean_invalid_actions"],
        "tool_success_rate": data["tool_success_rate"],
        "mean_response_tokens": data["mean_response_tokens"],
    }

arms = ("base", "outcome_reward", "process_reward", "quality_process_reward_v2")
report = {
    "protocol": {
        "dataset": "AIME-2024 (30 problems)",
        "max_new_tokens": 1024,
        "max_turns": 16,
        "temperature": 0.0,
        "seed": 20260825,
        "normalizes_markdown_code": True,
    },
    "8k_protocol_fix_four_arm": {arm: concise(summary(formal_root, arm)) for arm in arms},
    "16k_context_capacity_ablation": {arm: concise(summary(context_root, arm)) for arm in arms},
    "8k_v3_repaired_quality_reward": {
        arm: concise(summary(v3_root, arm)) for arm in ("base", "quality_process_reward_v3")
    },
    "notes": [
        "The 16K run is a capacity ablation of the existing checkpoints, not a replacement for the 8K formal protocol.",
        "quality_process_reward_v3 starts from the base model and changes the reward/answer protocol repair only; it does not resume v2.",
    ],
}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
  log "report written to $REPORT"
}

main() {
  printf 'started_at=%s\nsource_root=%s\n' "$(timestamp)" "$SOURCE_ROOT" > "$NIGHTLY_ROOT/RUN_INFO.txt"
  wait_for_formal_eval
  require_free_space
  start_v3_training
  # The Ray job intentionally reserves only GPUs 1--3.  This runs in parallel
  # on the otherwise unused inference GPU 0.
  run_context16k
  monitor_v3_training "$(< "$NIGHTLY_ROOT/v3-ray-job-id")"
  run_v3_eval
  write_report
  printf 'completed_at=%s\n' "$(timestamp)" > "$NIGHTLY_ROOT/COMPLETED"
  log "overnight follow-up complete"
}

main "$@"
