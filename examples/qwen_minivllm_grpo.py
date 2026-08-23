"""Real-model GRPO through the self-written stack:
mini-verl (RL loop) + mini-vllm engine (rollout backend) on Qwen3-0.6B.

Run on L20 with a GPU:
  python examples/qwen_minivllm_grpo.py --model .../Qwen3-0.6B-Base \
      --data .../training-2037-...parquet --limit 8 --iters 20
"""

from __future__ import annotations

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_verl.controller import Controller
from mini_verl.hf import HuggingFaceTrainerWorker, PromptExample
from mini_verl.protocol import Trajectory
from mini_verl.rollout_minivllm import MiniVllmRolloutWorker
from mini_verl.workers import RuleRewardWorker


def parse_final_answer(text: str) -> str:
    m = re.search(r"####\s*(-?[\d.,]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1].strip() if nums else ""


def normalize(n: str) -> str:
    try:
        return str(float(n))
    except ValueError:
        return n.strip()


def load_math_data(parquet_path: str, limit: int | None) -> tuple:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]
    prompts = []
    for i, row in enumerate(rows):
        messages = row["prompt"]
        if isinstance(messages, str):
            text = messages
        else:
            text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        gt = row.get("reward_model", {}).get("ground_truth", "")
        prompts.append(
            PromptExample(
                text + "\nPut your final answer after '####'.",
                {"group_id": f"p{i}", "expected_answer": str(gt)},
            )
        )
    return tuple(prompts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda")

    # self-written rollout backend: mini-vllm engine driving Qwen3
    from mini_vllm.engine import Engine
    from mini_vllm.transformers_adapter import TransformersAdapter

    adapter = TransformersAdapter(model)
    engine = Engine(adapter, temperature=1.0, top_p=0.9,
                    block_size=16, num_blocks=64,
                    max_prefill_tokens=128, max_running_tokens=512)

    prompts = load_math_data(args.data, args.limit)
    print(f"loaded {len(prompts)} prompts", flush=True)

    def reward(t: Trajectory) -> float:
        if t.response_text is None:
            return 0.0
        answer = parse_final_answer(t.response_text)
        expected = t.metadata["expected_answer"]
        return float(normalize(answer) == normalize(expected))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    controller = Controller(
        rollout_worker=MiniVllmRolloutWorker(
            engine=engine, tokenizer=tokenizer, prompts=prompts,
            group_size=args.group_size, max_new_tokens=args.max_new_tokens,
        ),
        reward_worker=RuleRewardWorker(reward),
        trainer_worker=HuggingFaceTrainerWorker(
            model=model, optimizer=optimizer,
            pad_token_id=int(tokenizer.pad_token_id),
        ),
    )

    for it in range(args.iters):
        result = controller.run_iteration()
        print(f"iter {it:3d} | reward={result.mean_reward:.3f} | "
              f"loss={result.metrics['loss']:.5f} | "
              f"clip={result.metrics['clip_fraction']:.3f}", flush=True)


if __name__ == "__main__":
    main()
