"""GRPO stability overnight ablation: reward clipping x entropy bonus.

Single-turn GRPO (no tools) with two stability interventions toggled
independently, so the four combinations form a 2x2 ablation:

  --clip-reward c : clamp group-relative advantages to [-c, c] before training
  --entropy-coef b: add -b * mean(log pi) entropy bonus to the GRPO loss

Baseline (clip=0, entropy=0) reproduces the plain GRPO setting of the 679 run.
Per-iteration reward / loss / clip_fraction / mean response length are written
to --out as JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_verl.controller import Controller
from mini_verl.hf import HuggingFaceTrainerWorker, PromptExample
from mini_verl.protocol import Trajectory, TrajectoryBatch, TrajectoryValidationError
from mini_verl.reward import apply_rewards, group_relative_advantages
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


def exact_answer_reward(trajectory: Trajectory) -> float:
    if trajectory.response_text is None:
        raise TrajectoryValidationError("reward requires response_text")
    expected = trajectory.metadata["expected_answer"]
    answer = parse_final_answer(trajectory.response_text)
    return float(normalize(answer) == normalize(expected))


class ClippedRewardWorker(RuleRewardWorker):
    """RuleRewardWorker with optional advantage clipping after normalization."""

    def __init__(self, reward_fn, clip: float = 0.0, epsilon: float = 1e-6):
        super().__init__(reward_fn, epsilon=epsilon)
        self.clip = clip

    def score(self, batch: TrajectoryBatch) -> TrajectoryBatch:
        scored = apply_rewards(batch, self.reward_fn)
        scored = group_relative_advantages(scored, epsilon=self.epsilon)
        if self.clip > 0:
            scored = TrajectoryBatch.from_iterable(
                trajectory.with_advantage(
                    max(-self.clip, min(self.clip, float(trajectory.advantage)))
                )
                for trajectory in scored.trajectories
            )
        return scored


class EntropyTrainerWorker(HuggingFaceTrainerWorker):
    """HuggingFaceTrainerWorker with an entropy bonus on the policy distribution.

    The bonus is -b * mean(log pi) over valid response tokens, computed from the
    logits the trainer already produces for the policy forward. This encourages
    exploration by penalizing confident (low-entropy) distributions.
    """

    def __init__(self, model, optimizer, pad_token_id, entropy_coef=0.0, **kwargs):
        super().__init__(model=model, optimizer=optimizer, pad_token_id=pad_token_id, **kwargs)
        self.entropy_coef = entropy_coef

    def train(self, batch: TrajectoryBatch, *, learner_policy_version: int):
        import torch as _torch
        from mini_verl.tensors import response_logprobs_from_logits
        if self.entropy_coef <= 0:
            return super().train(batch, learner_policy_version=learner_policy_version)
        # Reuse the parent's forward by doing a manual pass: replicate the
        # per-micro-batch loop to add the entropy term to each micro-loss.
        if learner_policy_version < 0:
            raise ValueError("learner_policy_version must be non-negative")
        if len(batch.policy_versions) != 1:
            raise TrajectoryValidationError("trainer requires one rollout policy version per batch")
        if any(trajectory.advantage is None for trajectory in batch.trajectories):
            raise TrajectoryValidationError("trainer requires advantages from a reward worker")
        from mini_verl.hf import causal_lm_inputs, batch_logprobs
        from mini_verl.algorithms.grpo import torch_grpo_loss
        device = next(self.model.parameters()).device
        self.model.train()
        packed_batches = self._training_batches(batch)
        total_response_tokens = sum(
            sum(trajectory.response_mask)
            for packed in packed_batches
            for trajectory in packed.batch.trajectories
        )
        if total_response_tokens <= 0:
            raise ValueError("response_mask must include at least one valid token")
        self.optimizer.zero_grad(set_to_none=True)
        metric_sums: dict[str, float] = {}
        entropy_sums = 0.0
        for packed in packed_batches:
            micro_batch = packed.batch
            micro_response_tokens = sum(sum(trajectory.response_mask) for trajectory in micro_batch.trajectories)
            inputs = causal_lm_inputs(micro_batch, pad_token_id=self.pad_token_id, device=device)
            logits = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).logits
            policy_scores = response_logprobs_from_logits(logits, micro_batch)
            old_logprobs = batch_logprobs(
                [trajectory.old_logprobs for trajectory in micro_batch.trajectories],
                device=device, dtype=policy_scores.values.dtype,
            )
            advantages = _torch.tensor(
                [float(trajectory.advantage) for trajectory in micro_batch.trajectories],
                dtype=policy_scores.values.dtype, device=device,
            )
            reference_logprobs = None
            if self.reference_model is not None:
                with _torch.inference_mode():
                    reference_logits = self.reference_model(
                        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
                    ).logits
                    reference_logprobs = response_logprobs_from_logits(reference_logits, micro_batch).values
            loss, metrics = torch_grpo_loss(
                policy_scores.values, old_logprobs, advantages, policy_scores.mask,
                reference_logprobs=reference_logprobs,
                clip_range=self.clip_range, beta=self.beta,
            )
            # Entropy bonus: -b * mean(log pi) over valid response tokens.
            logprobs = policy_scores.values
            mask = policy_scores.mask
            valid = mask.to(dtype=logprobs.dtype)
            entropy = -(logprobs * valid).sum() / valid.sum()
            loss = loss - self.entropy_coef * entropy
            entropy_sums += float(entropy.detach().item()) * micro_response_tokens
            (loss * (micro_response_tokens / total_response_tokens)).backward()
            for key, value in metrics.items():
                if key != "token_count":
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value.item()) * micro_response_tokens
        self.optimizer.step()
        real_sequence_tokens = sum(packed.real_sequence_tokens for packed in packed_batches)
        padded_sequence_tokens = sum(packed.padded_sequence_tokens for packed in packed_batches)
        aggregate = {key: value / total_response_tokens for key, value in metric_sums.items()}
        return {
            "loss": aggregate["policy_loss"] + self.beta * aggregate["kl_loss"]
                    - self.entropy_coef * (entropy_sums / total_response_tokens),
            **aggregate,
            "entropy": entropy_sums / total_response_tokens,
            "token_count": float(total_response_tokens),
            "train_microbatch_count": float(len(packed_batches)),
            "train_real_sequence_tokens": float(real_sequence_tokens),
            "train_padded_sequence_tokens": float(padded_sequence_tokens),
            "train_padding_ratio": (padded_sequence_tokens - real_sequence_tokens) / padded_sequence_tokens,
        }


class PlainRolloutWorker:
    """Single-turn batched rollout (no tools) — mirrors the 679-run setting.

    All group samples for a prompt are generated in one left-padded call, so
    G x batch rows share a single prefill+decode pass instead of G serial calls.
    """

    def __init__(self, model, tokenizer, prompts, group_size, max_new_tokens, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.device = device

    def _generate_batch(self, prompt_texts: list[str]) -> list[str]:
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompt_texts
        ]
        inp = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inp, max_new_tokens=self.max_new_tokens,
                do_sample=True, top_p=0.9, temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        results = []
        for i, row in enumerate(out):
            results.append(
                self.tokenizer.decode(row[inp["input_ids"].shape[1]:], skip_special_tokens=True)
            )
        return results

    def rollout(self, *, policy_version: int) -> TrajectoryBatch:
        trajectories: list[Trajectory] = []
        for prompt in self.prompts:
            texts = self._generate_batch([prompt.text] * self.group_size)
            for i, text in enumerate(texts):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--clip-reward", type=float, default=0.0,
                        help="0 = no advantage clipping; >0 = clamp advantage to [-c, c]")
    parser.add_argument("--entropy-coef", type=float, default=0.0,
                        help="0 = no entropy bonus; >0 = -b * mean(log pi)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="path to output JSON")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    prompts = load_math_data(args.data, args.limit)
    print(f"loaded {len(prompts)} prompts | clip={args.clip_reward} "
          f"entropy={args.entropy_coef}", flush=True)

    rollout = PlainRolloutWorker(
        model=model, tokenizer=tokenizer, prompts=prompts,
        group_size=args.group_size, max_new_tokens=args.max_new_tokens,
    )
    reward_worker = ClippedRewardWorker(exact_answer_reward, clip=args.clip_reward)
    trainer = EntropyTrainerWorker(
        model=model, optimizer=optimizer,
        pad_token_id=int(tokenizer.pad_token_id),
        entropy_coef=args.entropy_coef,
        train_micro_batch_size=8,
    )
    controller = Controller(
        rollout_worker=rollout,
        reward_worker=reward_worker,
        trainer_worker=trainer,
    )

    records: list[dict] = []
    for it in range(args.iters):
        t0 = time.time()
        result = controller.run_iteration()
        dt = time.time() - t0
        mean_len = result.response_token_count / result.trajectory_count
        record = {
            "iter": it,
            "mean_reward": result.mean_reward,
            "loss": result.metrics.get("loss"),
            "clip_fraction": result.metrics.get("clip_fraction"),
            "mean_response_tokens": mean_len,
            "entropy": result.metrics.get("entropy"),
            "seconds": round(dt, 3),
        }
        records.append(record)
        print(
            f"iter {it:3d} | reward={result.mean_reward:.3f} | "
            f"mean_len={mean_len:6.1f} | loss={record['loss']:.5f} | "
            f"clip={record['clip_fraction']:.3f} | {dt:.1f}s", flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "records": records}, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
