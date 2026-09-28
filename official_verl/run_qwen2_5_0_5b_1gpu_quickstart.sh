#!/usr/bin/env bash
# Upstream's published quickstart, copied nearly verbatim:
#   verl docs/start/quickstart.rst — "Quickstart: PPO training on GSM8K dataset"
#   (PPO, Qwen2.5-0.5B-Instruct, FSDP actor+critic, vLLM rollout, one GPU)
#
# Every algorithm and resource field below is unchanged from the published
# command. The deviations are exactly four, all recorded in the run log:
# a single visible GPU, caller-supplied data paths, MODEL_PATH, and
# trainer.total_epochs=1 (upstream ships 15 = 435 steps ≈ 4.5 h).
#
# Requires the pinned checkout (bootstrap_verl.sh + VERL_REVISION=v0.7.1) and
# the official-schema parquet pair (official_verl/prepare_openai_gsm8k.py).
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${VERL_DIR:?Set VERL_DIR to the pinned official verl checkout.}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the official-schema GSM8K train parquet.}"
: "${TEST_FILE:?Set TEST_FILE to the official-schema GSM8K test parquet.}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-verl-quickstart-qwen2.5-0.5b-1gpu}
RUN_ROOT=${RUN_ROOT:-"$(pwd)/artifacts/$EXPERIMENT_NAME"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
# hf-mirror 302-redirects Xet-backed weights to a host it does not mirror, which
# stalls the large safetensors download. Prefer the real endpoint via a proxy.
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
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
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
