#!/usr/bin/env bash
# Two-step Qwen3.5-4B compatibility and reward calibration; not a quality run.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${VERL_DIR:?Set VERL_DIR to the pinned official verl checkout.}"
: "${MODEL_PATH:?Set MODEL_PATH to the complete Qwen3.5-4B snapshot.}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the 2,048-row OpenR1 calibration parquet.}"
: "${TEST_FILE:?Set TEST_FILE to the frozen 64-row MATH validation parquet.}"
PYTHON_BIN=${PYTHON_BIN:-python}
COMPAT_PATH=${COMPAT_PATH:-"$SCRIPT_DIR/compat"}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}

if (( MAX_PROMPT_LENGTH <= 0 || MAX_RESPONSE_LENGTH <= 0 || MAX_MODEL_LEN < MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH )); then
  echo "invalid sequence limits: prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH model=$MAX_MODEL_LEN" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONPATH="$COMPAT_PATH${PYTHONPATH:+:$PYTHONPATH}"
# The container allows CUDA work but blocks CUDA-IPC handle duplication
# (pidfd_getfd).  The opt-in compat patch selects verl's existing CPU-buffer
# transfer path and backs its large bucket with a temporary mmap file rather
# than the container's 1 GiB /dev/shm mount.
export MINI_VERL_FORCE_MMAP_WEIGHT_TRANSFER=${MINI_VERL_FORCE_MMAP_WEIGHT_TRANSFER:-1}
export MINI_VERL_WEIGHT_TRANSFER_MMAP_DIR=${MINI_VERL_WEIGHT_TRANSFER_MMAP_DIR:-/tmp}
PYTORCH_CUDA_ALLOC_CONF_VALUE=${PYTORCH_CUDA_ALLOC_CONF_VALUE:-}
if [[ -n "$PYTORCH_CUDA_ALLOC_CONF_VALUE" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF_VALUE"
fi

RUN_ROOT=${RUN_ROOT:-"$(pwd)/artifacts/qwen3.5-4b-openr1-grpo-calibration"}
PROJECT_NAME=${PROJECT_NAME:-official-verl-grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3.5-4b-openr1-calibration}
TRAINING_STEPS=${TRAINING_STEPS:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
ROLLOUT_N=${ROLLOUT_N:-4}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-8}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-false}
TRAINER_GPUS=${TRAINER_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
ROLLOUT_TP=${ROLLOUT_TP:-2}
SAVE_FREQ=${SAVE_FREQ:-1}
TEST_FREQ=${TEST_FREQ:-1}

if (( TRAINING_STEPS <= 0 || TRAIN_BATCH_SIZE <= 0 || PPO_MINI_BATCH_SIZE <= 0 || ROLLOUT_N <= 0 || AGENT_NUM_WORKERS <= 0 || TRAINER_GPUS <= 0 || ROLLOUT_GPUS <= 0 || ROLLOUT_TP <= 0 || SAVE_FREQ <= 0 || TEST_FREQ <= 0 )); then
  echo "training, rollout, worker, and topology values must be positive" >&2
  exit 2
fi

if (( TRAINER_GPUS + ROLLOUT_GPUS != 4 )); then
  echo "TRAINER_GPUS + ROLLOUT_GPUS must equal the four visible GPUs: trainer=$TRAINER_GPUS rollout=$ROLLOUT_GPUS" >&2
  exit 2
fi

if (( ROLLOUT_GPUS % ROLLOUT_TP != 0 )); then
  echo "ROLLOUT_GPUS must be divisible by ROLLOUT_TP: rollout=$ROLLOUT_GPUS tp=$ROLLOUT_TP" >&2
  exit 2
fi

# Upstream validates this later, after Ray and model startup.  Reject it here
# so a 3-trainer-GPU experiment reports its incompatible global batch early.
if (((TRAIN_BATCH_SIZE * ROLLOUT_N) % TRAINER_GPUS != 0)); then
  echo "TRAIN_BATCH_SIZE * ROLLOUT_N must be divisible by TRAINER_GPUS: batch=$TRAIN_BATCH_SIZE n=$ROLLOUT_N trainer_gpus=$TRAINER_GPUS" >&2
  exit 2
fi

if (( PPO_MINI_BATCH_SIZE > TRAIN_BATCH_SIZE )); then
  echo "PPO_MINI_BATCH_SIZE must not exceed TRAIN_BATCH_SIZE: mini=$PPO_MINI_BATCH_SIZE batch=$TRAIN_BATCH_SIZE" >&2
  exit 2
fi

# The upstream agent loop splits the generated sequences equally across its
# CPU workers. A GRPO update has TRAIN_BATCH_SIZE * ROLLOUT_N sequences, so
# reject a topology which would otherwise fail only after costly model startup.
if (((TRAIN_BATCH_SIZE * ROLLOUT_N) % AGENT_NUM_WORKERS != 0)); then
  echo "TRAIN_BATCH_SIZE * ROLLOUT_N must be divisible by AGENT_NUM_WORKERS: batch=$TRAIN_BATCH_SIZE n=$ROLLOUT_N workers=$AGENT_NUM_WORKERS" >&2
  exit 2
fi

# This is a read-only safety check. It never kills or changes another process.
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "refusing to launch: one or more selected GPUs has a compute process" >&2
  nvidia-smi >&2 || true
  exit 3
fi

"$PYTHON_BIN" "$SCRIPT_DIR/preflight.py" \
  --verl-dir "$VERL_DIR" --require-cuda --require-runtime --require-lock

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/logs" "$RUN_ROOT/rollout_samples" "$RUN_ROOT/validation_samples"
cd "$VERL_DIR"
set +e
# This is a resource-bounded compatibility calibration, not a throughput or
# quality recipe. Keep a small number of prompts per update with four GRPO
# samples each.
# A 512-token response completed the first update but OOMed in the second
# backward.  The current default is therefore 384 response tokens (896 context)
# with an explicitly short-solution prompt contract. A no-update diagnostic on
# real OpenR1 rows found a non-degenerate four-sample group [0, 0, 0, 1] under
# that contract; earlier 256-token and final-only alternatives were reward
# degenerate. The limits are parameterized for controlled experiments;
# change only one limit at a time and give every attempt a fresh RUN_ROOT.
# FSDP2 policy and optimizer-state offload, plus
# non-foreach AdamW, avoid the separate policy-parameter and AdamW temporary
# peaks observed on a pair of L20s. The L20 pair also cannot initialize vLLM's
# optional custom all-reduce kernel, so select its NCCL fallback. A 3 GiB mmap
# transfer bucket accommodates the model's 2.54 GiB FP32 embedding tensor.
"$PYTHON_BIN" -m verl.experimental.one_step_off_policy.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TEST_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.shuffle=False \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
  critic.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.offload_policy=True \
  '+actor_rollout_ref.actor.optim.override_optimizer_config={foreach: false}' \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_NUM_WORKERS" \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.val_before_train=True \
  trainer.logger='[console,tensorboard]' \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$RUN_ROOT/checkpoints" \
  trainer.rollout_data_dir="$RUN_ROOT/rollout_samples" \
  trainer.validation_data_dir="$RUN_ROOT/validation_samples" \
  trainer.log_val_generations=10 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$TRAINING_STEPS" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="$TRAINER_GPUS" \
  rollout.nnodes=1 \
  rollout.n_gpus_per_node="$ROLLOUT_GPUS" \
  "$@" 2>&1 | tee "$RUN_ROOT/logs/train.log"
launch_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$launch_status" > "$RUN_ROOT/logs/exit_status"
exit "$launch_status"
