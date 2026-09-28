#!/usr/bin/env bash
# Same published quickstart command as the 1-GPU launcher, spread over four GPUs.
#
# Three differences from run_qwen2_5_0_5b_1gpu_quickstart.sh, and one addition:
#   1. trainer.n_gpus_per_node=4 and no CUDA_VISIBLE_DEVICES pin.
#   2. trainer.experiment_name defaults to a distinct value, so the default
#      resume_mode=auto cannot pick up the 1-GPU run's checkpoints.
#   3. trainer.val_before_train=True. Upstream ships False; the 1-GPU run
#      inherited that and produced no base-model anchor, which makes its rising
#      validation curve uninterpretable. This is the control, not a knob.
#
# Note upstream's own sizing advice is NOT applied: it suggests
# ppo_micro_batch_size_per_gpu=1 and critic.ppo_micro_batch_size_per_gpu=1 for
# HBM < 32 GB, but 4 fits comfortably in 24 GB (headroom is reported in the run
# log), so the published batch sizes are preserved.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${VERL_DIR:?Set VERL_DIR to the pinned official verl checkout.}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the official-schema GSM8K train parquet.}"
: "${TEST_FILE:?Set TEST_FILE to the official-schema GSM8K test parquet.}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-verl-quickstart-qwen2.5-0.5b-4gpu}
RUN_ROOT=${RUN_ROOT:-"$(pwd)/artifacts/$EXPERIMENT_NAME"}

export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
unset HF_ENDPOINT

mkdir -p "$RUN_ROOT/logs"

cd "$VERL_DIR"
set +e
python -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TEST_FILE" \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  critic.optim.lr=1e-5 \
  critic.model.path="$MODEL_PATH" \
  critic.ppo_micro_batch_size_per_gpu=4 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=console \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  "$@" 2>&1 | tee "$RUN_ROOT/logs/train.log"
launch_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$launch_status" > "$RUN_ROOT/logs/exit_status"
exit "$launch_status"
