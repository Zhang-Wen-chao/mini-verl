#!/usr/bin/env bash
# Read-only evaluation of Strategy 2 checkpoints on held-out AIME-2024.
# Run inside the slime-dev container; it never updates the source checkpoints.
# The evaluator reports Markdown-fence normalization independently, so a
# recoverable formatting drift is not confused with a sandbox rejection.  This
# launcher evaluates a capability-tolerant mode; omit its final normalization
# flag to reproduce the strict training protocol.
set -euo pipefail

BASE=/mnt/storage01/zhangwenchao02
SCRIPT_DIR=$BASE/experiments/strategy2
RUN_ROOT=${RUN_ROOT:-$BASE/strategy2-eval-$(date +%Y%m%dT%H%M%S)}
DATA=$BASE/data/aime-2024/aime-2024.jsonl
ORIGIN_HF=$BASE/models/Qwen3-4B-Instruct-2507
MAX_PROBLEMS=${MAX_PROBLEMS:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS:-4096}
MAX_TURNS=${MAX_TURNS:-16}
TEMPERATURE=${TEMPERATURE:-0.0}
TOOL_TIMEOUT_SECONDS=${TOOL_TIMEOUT_SECONDS:-20}

mkdir -p "$RUN_ROOT/models" "$RUN_ROOT/results"

convert_checkpoint() {
  local arm=$1
  local input_dir=$BASE/retool-rl-smoke/$arm/ckpt/iter_0000019
  local output_dir=$RUN_ROOT/models/$arm-hf
  python3 /root/slime/tools/convert_torch_dist_to_hf.py \
    --input-dir "$input_dir" \
    --output-dir "$output_dir" \
    --origin-hf-dir "$ORIGIN_HF" \
    --vocab-size 151936
}

run_eval() {
  local name=$1
  local model=$2
  PYTHONPATH=$SCRIPT_DIR:/root/slime CUDA_VISIBLE_DEVICES=0 \
    python3 $SCRIPT_DIR/evaluate_retool.py \
      --model "$model" \
      --data "$DATA" \
      --output-dir "$RUN_ROOT/results/$name" \
      --max-problems "$MAX_PROBLEMS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --max-context-tokens "$MAX_CONTEXT_TOKENS" \
      --max-turns "$MAX_TURNS" \
      --temperature "$TEMPERATURE" \
      --tool-timeout-seconds "$TOOL_TIMEOUT_SECONDS" \
      --normalize-markdown-code \
      --seed 20260825
}

run_eval base "$ORIGIN_HF"
convert_checkpoint outcome_reward
run_eval outcome_reward "$RUN_ROOT/models/outcome_reward-hf"
convert_checkpoint process_reward
run_eval process_reward "$RUN_ROOT/models/process_reward-hf"

python3 - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for summary in sorted(root.glob("results/*/summary.json")):
    data = json.loads(summary.read_text())
    print(
        summary.parent.name,
        f"accuracy={data['accuracy']:.3f}",
        f"tool_use_rate={data['tool_use_rate']:.3f}",
        f"mean_tool_calls={data['mean_tool_calls']:.3f}",
        f"tool_success_rate={data['tool_success_rate']:.3f}",
        f"terminal={data['terminal_status_counts']}",
    )
PY
