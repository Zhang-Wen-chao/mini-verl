#!/usr/bin/env bash
# A deliberately small adaptation of upstream's
# verl/experimental/one_step_off_policy/shell/grpo_0.6b_gsm8k_fsdp2_2_6.sh.
# It preserves the official main entrypoint and GRPO/FSDP2/vLLM architecture,
# while fitting the available 4 x L20 topology: 2 trainer GPUs + 2 rollout GPUs.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${VERL_DIR:?Set VERL_DIR to the pinned official verl checkout.}"
: "${MODEL_PATH:?Set MODEL_PATH to a complete Qwen3-0.6B snapshot.}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the 256-row official-schema parquet.}"
: "${TEST_FILE:?Set TEST_FILE to the 64-row official-schema parquet.}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

RUN_ROOT=${RUN_ROOT:-"$(pwd)/artifacts/qwen3-0.6b-gsm8k-grpo-4gpu-smoke"}
PROJECT_NAME=${PROJECT_NAME:-official-verl-grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-0.6b-gsm8k-4gpu-smoke}

# Validate the exact interpreter before Ray can reserve the four visible GPUs.
# This catches, for example, a host-only virtualenv whose Python target is not
# mounted in this container.
python "$SCRIPT_DIR/preflight.py" \
  --verl-dir "$VERL_DIR" --require-cuda --require-runtime --require-lock

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/logs"

cd "$VERL_DIR"
# Preserve the training exit code while streaming stdout/stderr to a durable log.
set +e
python -m verl.experimental.one_step_off_policy.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TEST_FILE" \
  data.train_batch_size=16 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
  critic.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.val_before_train=True \
  trainer.logger='[console,tensorboard]' \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$RUN_ROOT/checkpoints" \
  trainer.save_freq=1 \
  trainer.test_freq=1 \
  trainer.total_epochs=1 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=2 \
  rollout.nnodes=1 \
  rollout.n_gpus_per_node=2 \
  "$@" 2>&1 | tee "$RUN_ROOT/logs/train.log"
launch_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$launch_status" > "$RUN_ROOT/logs/exit_status"
exit "$launch_status"
