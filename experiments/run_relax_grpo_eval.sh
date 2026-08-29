#!/usr/bin/env bash
# Evaluate one HF-format Relax checkpoint without updating its weights.
# Run inside the slime-dev container against the existing local Ray cluster.
set -euo pipefail

BASE=${BASE:-/mnt/storage01/zhangwenchao02}
RELAX_ROOT=${RELAX_ROOT:-$BASE/repos/Relax}
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
MODEL=${MODEL:?MODEL must point to an HF-format model directory}
NAME=${NAME:?NAME identifies this evaluation arm}
OUTPUT_ROOT=${OUTPUT_ROOT:-$BASE/evals/relax-qwen3-4b-400-20260829/results_4096}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-4096}
ROLLOUT_SEED=${ROLLOUT_SEED:-20260829}

TRAIN_DATA=$BASE/data/dapo-math-17k/dapo-math-17k.jsonl
EVAL_DATA=$BASE/data/aime-2024/aime-2024.jsonl
RESULT_DIR=$OUTPUT_ROOT/$NAME
LOG_DIR=$OUTPUT_ROOT/logs

mkdir -p "$RESULT_DIR" "$LOG_DIR"

runtime_env=$(python3 - "$RELAX_ROOT" "$MEGATRON_ROOT" <<'PY'
import json
import sys

relax_root, megatron_root = sys.argv[1:]
print(json.dumps({
    "env_vars": {
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{relax_root}:{megatron_root}",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "OPENBLAS_NUM_THREADS": "16",
        "NCCL_NVLS_ENABLE": "0",
    }
}))
PY
)

# Relax currently requires num_rollout > 0. Debug-rollout-only therefore runs
# the startup AIME evaluation followed by one disposable four-prompt rollout;
# neither path creates an actor or performs an optimizer update.
ray job submit --no-wait \
  --address=http://127.0.0.1:8265 \
  --working-dir "$RELAX_ROOT" \
  --runtime-env-json "$runtime_env" \
  -- python3 -m relax.entrypoints.train \
  --debug-rollout-only \
  --resource '{"rollout": [1, 4]}' \
  --hf-checkpoint "$MODEL" \
  --prompt-data "$TRAIN_DATA" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rm-type dapo \
  --reward-key score \
  --num-rollout 1 \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len "$MAX_RESPONSE_LEN" \
  --rollout-temperature 0.8 \
  --rollout-seed "$ROLLOUT_SEED" \
  --global-batch-size 8 \
  --eval-interval 1 \
  --eval-prompt-data aime "$EVAL_DATA" \
  --eval-input-key prompt \
  --eval-label-key label \
  --n-samples-per-eval-prompt 8 \
  --eval-temperature 0.8 \
  --eval-max-response-len "$MAX_RESPONSE_LEN" \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.8 \
  --rollout-result-dir "$RESULT_DIR" \
  --no-use-metrics-service \
  --no-use-tensorboard \
  >"$LOG_DIR/$NAME-submit.log"

cat "$LOG_DIR/$NAME-submit.log"
