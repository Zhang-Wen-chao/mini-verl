#!/bin/bash
# Strategy 2: outcome, legacy process, and repaired quality-process rewards.
# Usage: QUALITY_PROCESS_REWARD=1 bash reward_compare.sh  (repaired arm)
#        PROCESS_REWARD=1 bash reward_compare.sh          (legacy ablation)
#        bash reward_compare.sh                            (outcome arm, default)
# Launches retool GRPO with the strategy2/ generate_with_retool.py
# (reward_func honors PROCESS_REWARD).
set -ex
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1,2,3

# Which arm are we?
if [ -n "${QUALITY_PROCESS_REWARD:-}" ]; then
  export QUALITY_PROCESS_REWARD=1
  export PROCESS_REWARD=""
  ARM_BASE="quality_process_reward"
elif [ -n "${PROCESS_REWARD:-}" ]; then
  export PROCESS_REWARD=1
  export QUALITY_PROCESS_REWARD=""
  ARM_BASE="process_reward"
else
  export PROCESS_REWARD=""
  export QUALITY_PROCESS_REWARD=""
  ARM_BASE="outcome_reward"
fi

# A suffix isolates a repaired rerun from every existing arm. Requiring an
# underscore makes accidental path replacement conspicuous and prevents
# overwriting an old checkpoint directory.
RUN_SUFFIX=${RUN_SUFFIX:-}
case "$RUN_SUFFIX" in
  ""|_[A-Za-z0-9_-]*) ;;
  *) echo "RUN_SUFFIX must be empty or start with '_'" >&2; exit 2 ;;
esac
ARM="${ARM_BASE}${RUN_SUFFIX}"
echo "=== ARM: $ARM ==="

BASE=/mnt/storage01/zhangwenchao02
MODEL_HF=$BASE/models/Qwen3-4B-Instruct-2507
MODEL_MC=$BASE/models/Qwen3-4B-Instruct-2507_torch_dist
SAVE_DIR=$BASE/retool-rl-smoke/${ARM}
DAPO_DATA=$BASE/data/dapo-math-17k/dapo-math-17k.jsonl
SCRIPT_DIR=$BASE/experiments/strategy2
mkdir -p "$SAVE_DIR"

source /root/slime/scripts/models/qwen3-4B.sh

CKPT_ARGS=(
   --hf-checkpoint $MODEL_HF
   --ref-load $MODEL_MC
   --save $SAVE_DIR/ckpt/
   --save-interval 100
   --rotary-base 5000000
)

ROLLOUT_ARGS=(
   --prompt-data $DAPO_DATA
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout 20
   --rollout-batch-size 2
   --n-samples-per-prompt 4
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 8
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

# ---- launch or safely reuse Ray, then submit ----
# The original launcher reset every local Ray/SGLang process. That is useful
# only when deliberately replacing a stale private cluster; it is unsafe on a
# shared host. REUSE_EXISTING_RAY=1 validates and reuses the current cluster.
if [ "${REUSE_EXISTING_RAY:-}" = "1" ]; then
  ray status
else
  pkill -9 sglang 2>/dev/null || true
  sleep 2
  ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
  sleep 2

  export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
  ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 3 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"PROCESS_REWARD\": \"${PROCESS_REWARD}\",
    \"QUALITY_PROCESS_REWARD\": \"${QUALITY_PROCESS_REWARD}\",
    \"RETOOL_SYNC_CKPT_STAGING\": \"${RETOOL_SYNC_CKPT_STAGING:-0}\"
  }
}"

cd /root/slime
RAY_JOB_ARGS=()
if [ "${RAY_SUBMIT_NO_WAIT:-}" = "1" ]; then
  RAY_JOB_ARGS+=(--no-wait)
fi

ray job submit "${RAY_JOB_ARGS[@]}" --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
