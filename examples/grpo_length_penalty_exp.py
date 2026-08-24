"""Length-inflation fix experiment: does a length penalty stop response length
growth without hurting accuracy?

Control:  reward = 1.0 if answer correct else 0.0   (identical to 679 run)
Treated:  reward = correct_credit - lambda * max(0, len - target)
          where len is response token count, target the expected length.

Prints per-iteration mean response length and accuracy so the two groups can be
compared directly. Small scale (few prompts, tens of iters) is enough to show
whether the penalty bends the length curve.
"""

from __future__ import annotations

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_verl.controller import Controller
from mini_verl.hf import HuggingFaceTrainerWorker, PromptExample
from mini_verl.protocol import Trajectory, TrajectoryBatch, TrajectoryValidationError
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


def load_math_data(parquet_path: str, limit: int | None = None) -> tuple[PromptExample, ...]:
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


class PlainRolloutWorker:
    """Single-turn rollout (no tools) — mirrors the 679-run plain GRPO setting."""

    def __init__(self, model, tokenizer, prompts, group_size, max_new_tokens, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.device = device

    def _generate(self, prompt_text: str) -> str:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inp = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inp,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

    def rollout(self, *, policy_version: int) -> TrajectoryBatch:
        trajectories: list[Trajectory] = []
        for prompt in self.prompts:
            for i in range(self.group_size):
                text = self._generate(prompt.text)
                tokens = self.tokenizer.encode(text)
                trajectories.append(
                    Trajectory(
                        prompt_token_ids=self.tokenizer.encode(prompt.text),
                        response_token_ids=tokens,
                        old_logprobs=(0.0,) * len(tokens),
                        policy_version=policy_version,
                        group_id=prompt.metadata["group_id"],
                        response_text=text,
                        prompt_text=prompt.text,
                        metadata={**prompt.metadata, "sample_index": i},
                    )
                )
        return TrajectoryBatch.from_iterable(trajectories)


def make_reward(length_penalty: float, target_length: int):
    """Return a reward fn. length_penalty=0 reproduces the control (679) reward."""

    def reward(trajectory: Trajectory) -> float:
        if trajectory.response_text is None:
            raise TrajectoryValidationError("reward requires response_text")
        expected = trajectory.metadata["expected_answer"]
        answer = parse_final_answer(trajectory.response_text)
        correct = float(normalize(answer) == normalize(expected))
        if length_penalty <= 0:
            return correct
        excess = max(0, len(trajectory.response_token_ids) - target_length)
        return correct - length_penalty * excess

    return reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--length-penalty", type=float, default=0.0,
                        help="0 = control (no penalty); >0 = penalize excess length")
    parser.add_argument("--target-length", type=int, default=64,
                        help="length below which no penalty applies")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    prompts = load_math_data(args.data, args.limit)
    print(f"loaded {len(prompts)} prompts | penalty={args.length_penalty} "
          f"target_len={args.target_length}", flush=True)

    rollout = PlainRolloutWorker(
        model=model, tokenizer=tokenizer, prompts=prompts,
        group_size=args.group_size, max_new_tokens=args.max_new_tokens,
    )
    reward_fn = make_reward(args.length_penalty, args.target_length)
    controller = Controller(
        rollout_worker=rollout,
        reward_worker=RuleRewardWorker(reward_fn),
        trainer_worker=HuggingFaceTrainerWorker(
            model=model, optimizer=optimizer,
            pad_token_id=int(tokenizer.pad_token_id),
        ),
    )

    for it in range(args.iters):
        result = controller.run_iteration()
        mean_len = result.response_token_count / result.trajectory_count
        print(f"iter {it:3d} | reward={result.mean_reward:.3f} | "
              f"mean_len={mean_len:6.1f} | "
              f"loss={result.metrics['loss']:.5f} | "
              f"clip={result.metrics['clip_fraction']:.3f}", flush=True)


if __name__ == "__main__":
    main()
