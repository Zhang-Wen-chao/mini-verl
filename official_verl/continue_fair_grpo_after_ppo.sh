#!/usr/bin/env bash
# Start the matching GRPO leg only after an already-running PPO fair-comparison
# leg has passed its artifact and numerical-health gate.  Run this inside the
# same CUDA container as the child launchers. This supervisor never manipulates
# GPU processes; each child runner performs its own idle-GPU check.
set -euo pipefail

: "${ROOT:?Set ROOT to the persistent mini-verl-l20 checkout.}"
: "${PPO_RUN_ROOT:?Set PPO_RUN_ROOT to the already-started PPO artifact directory.}"
TRAINING_STEPS=${TRAINING_STEPS:-5}
SEED=${SEED:-42}
WAIT_SECONDS=${WAIT_SECONDS:-43200}
POLL_SECONDS=${POLL_SECONDS:-30}
GRPO_RUN_ID=${GRPO_RUN_ID:-"qwen3.5-4b-fair-grpo-${TRAINING_STEPS}step-seed${SEED}-$(date +%Y%m%dT%H%M%S)"}
GRPO_RUN_ROOT=${GRPO_RUN_ROOT:-"$ROOT/artifacts/$GRPO_RUN_ID"}
PPO_GATE_LOG="$PPO_RUN_ROOT/logs/gate.log"
PPO_FAILURE="$PPO_RUN_ROOT/logs/gate_failure"
SUPERVISOR_LOG="$GRPO_RUN_ROOT/logs/supervisor.log"

(( TRAINING_STEPS > 0 && WAIT_SECONDS > 0 && POLL_SECONDS > 0 )) || {
  echo "TRAINING_STEPS, WAIT_SECONDS, and POLL_SECONDS must be positive" >&2
  exit 2
}
mkdir -p "$GRPO_RUN_ROOT/logs"
log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$SUPERVISOR_LOG"; }
stop() { log "SUPERVISOR_STOPPED: $*"; printf '%s\n' "$*" >"$GRPO_RUN_ROOT/logs/supervisor_failure"; exit 1; }

deadline=$(( $(date +%s) + WAIT_SECONDS ))
log "waiting for PPO fair leg to pass before starting matching GRPO; no GPU processes will be changed"
while true; do
  [[ -f "$PPO_FAILURE" ]] && stop "PPO gate failed: $(tr '\n' ' ' <"$PPO_FAILURE")"
  if [[ -f "$PPO_GATE_LOG" ]] && grep -q "FAIR_COMPARISON_LEG_FINISHED: ppo completed ${TRAINING_STEPS} finite updates" "$PPO_GATE_LOG"; then
    break
  fi
  (( $(date +%s) < deadline )) || stop "timed out waiting for PPO fair leg"
  sleep "$POLL_SECONDS"
done

log "PPO passed its gate; starting the matching GRPO leg under the same shared rollout contract"
env ROOT="$ROOT" ALGORITHM=grpo TRAINING_STEPS="$TRAINING_STEPS" SEED="$SEED" \
  RUN_ID="$GRPO_RUN_ID" RUN_ROOT="$GRPO_RUN_ROOT" WAIT_SECONDS="$WAIT_SECONDS" \
  /bin/bash "$ROOT/official_verl/await_fair_algorithm_comparison.sh"
log "FAIR_COMPARISON_PAIR_FINISHED: PPO and GRPO both passed their gates"
