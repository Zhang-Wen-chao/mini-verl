#!/bin/bash
# Overnight experiment matrix for mini-verl agent RL (runs on l20, in zhangwenchao-megatron container)
# Serial on GPU 0 for comparability. Each run writes JSON + log.
set -u

cd /mnt/storage01/zhangwenchao02/repos/mini-verl-l20
export PYTHONPATH=/mnt/storage01/zhangwenchao02/repos/mini-verl-l20
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL=.official-verl/models/Qwen3-0.6B
DATA=.official-verl/data/gsm8k-smoke/train.parquet
OUT=/mnt/storage01/zhangwenchao02/repos/mini-verl-l20/experiments/results
mkdir -p "$OUT"

echo "=== overnight matrix start $(date) ==="

run_agent() {
  local name=$1; shift
  echo "--- agent:$name $(date) ---"
  python examples/agent_rl_overnight.py \
    --model "$MODEL" --data "$DATA" --out "$OUT/agent_${name}.json" \
    "$@" 2>&1 | tee "$OUT/agent_${name}.log"
  echo "--- done agent:$name $(date) rc=$? ---"
}

run_stab() {
  local name=$1; shift
  echo "--- stab:$name $(date) ---"
  python examples/grpo_stability_overnight.py \
    --model "$MODEL" --data "$DATA" --out "$OUT/stab_${name}.json" \
    "$@" 2>&1 | tee "$OUT/stab_${name}.log"
  echo "--- done stab:$name $(date) rc=$? ---"
}

# ---- Agent mainline: tool vs no-tool x 3 reward versions ----
# 1. no-tool control (mirrors 679 run)
run_agent notool_finalonly --mode no-tool --reward-version final-only \
  --limit 6 --iters 20 --group-size 4 --max-new-tokens 512 --lr 1e-6 --seed 42

# 2. tool + final-only (core question: does tool availability help?)
run_agent tool_finalonly --mode tool --reward-version final-only \
  --limit 6 --iters 20 --group-size 4 --max-new-tokens 512 --lr 1e-6 --seed 42

# 3. tool + tool-bonus (reward shaping: reward successful tool use)
run_agent tool_toolbonus --mode tool --reward-version tool-bonus \
  --limit 6 --iters 20 --group-size 4 --max-new-tokens 512 --lr 1e-6 --seed 42

# 4. tool + process (step-wise credit vs sparse final reward)
run_agent tool_process --mode tool --reward-version process \
  --limit 6 --iters 20 --group-size 4 --max-new-tokens 512 --lr 1e-6 --seed 42

# ---- Stability ablation: clip-reward x entropy-coef (single-turn) ----
# 5. baseline
run_stab clip0_ent0  --limit 6 --iters 30 --group-size 4 --max-new-tokens 512 --lr 1e-6 --seed 42 --clip-reward 0 --entropy-coef 0
# 6. reward clipping only
run_stab clip3_ent0   --limit 6 --iters 30 --group-size 4 --max-new-tokens 512 --lr 1e-6 --seed 42 --clip-reward 3 --entropy-coef 0
# 7. entropy bonus only
run_stab clip0_ent1e2 --limit 6 --iters 30 --group-size 4 --max-new-tokens 512 --lr 1e-6 --seed 42 --clip-reward 0 --entropy-coef 0.01
# 8. both
run_stab clip3_ent1e2 --limit 6 --iters 30 --group-size 4 --max-new-tokens 512 --lr 1e-6 --seed 42 --clip-reward 3 --entropy-coef 0.01

echo "=== overnight matrix done $(date) ==="
