"""An end-to-end GRPO iteration on a tiny categorical language-policy task.

This intentionally avoids model downloads. Each prompt has a correct one-token
answer. The policy generates G responses, scores them with the same rule-reward
boundary used by a real LLM rollout, normalizes group advantages, and applies the
token-level GRPO objective. It is a correctness/demo workload, not a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch

from mini_verl.algorithms.grpo import torch_grpo_loss
from mini_verl.config import RunConfig, seed_everything
from mini_verl.controller import Controller
from mini_verl.protocol import Trajectory, TrajectoryBatch
from mini_verl.workers import RuleRewardWorker


@dataclass(frozen=True, slots=True)
class ToyRunResult:
    initial_pass_at_1: float
    final_pass_at_1: float
    final_reward: float
    final_loss: float
    final_iteration_seconds: float


def _answer_reward(trajectory: Trajectory) -> float:
    return float(trajectory.response_token_ids[0] == trajectory.metadata["correct_token"])


def _sample_batch(
    logits: torch.Tensor,
    correct_tokens: torch.Tensor,
    *,
    group_size: int,
    policy_version: int,
) -> TrajectoryBatch:
    probabilities = torch.softmax(logits.detach(), dim=-1)
    trajectories: list[Trajectory] = []
    for prompt_id, row in enumerate(probabilities):
        sampled = torch.multinomial(row, num_samples=group_size, replacement=True)
        for sample_index, action in enumerate(sampled.tolist()):
            trajectories.append(
                Trajectory(
                    prompt_token_ids=(prompt_id,),
                    response_token_ids=(action,),
                    old_logprobs=(float(torch.log(row[action]).item()),),
                    policy_version=policy_version,
                    group_id=f"prompt-{prompt_id}",
                    metadata={"correct_token": int(correct_tokens[prompt_id]), "sample_index": sample_index},
                )
            )
    return TrajectoryBatch.from_iterable(trajectories)


def _pass_at_1(logits: torch.Tensor, correct_tokens: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == correct_tokens).float().mean().item())


@dataclass(slots=True)
class ToyRolloutWorker:
    """A stand-in for an LLM generation backend."""

    logits: torch.Tensor
    correct_tokens: torch.Tensor
    group_size: int

    def rollout(self, *, policy_version: int) -> TrajectoryBatch:
        return _sample_batch(
            self.logits,
            self.correct_tokens,
            group_size=self.group_size,
            policy_version=policy_version,
        )


@dataclass(slots=True)
class ToyTrainerWorker:
    """A trainer backend that consumes the framework's trajectory contract."""

    logits: torch.nn.Parameter
    optimizer: torch.optim.Optimizer
    device: torch.device

    def train(self, batch: TrajectoryBatch, *, learner_policy_version: int) -> Mapping[str, float]:
        if learner_policy_version < 0:
            raise ValueError("learner_policy_version must be non-negative")
        if len(batch.policy_versions) != 1:
            raise ValueError("trainer requires one rollout policy version per batch")
        prompt_ids = torch.tensor(
            [trajectory.prompt_token_ids[0] for trajectory in batch.trajectories], device=self.device
        )
        action_ids = torch.tensor(
            [trajectory.response_token_ids[0] for trajectory in batch.trajectories], device=self.device
        )
        old_logprobs = torch.tensor(
            [[trajectory.old_logprobs[0]] for trajectory in batch.trajectories], device=self.device
        )
        advantages = torch.tensor(
            [trajectory.advantage for trajectory in batch.trajectories], device=self.device
        )
        new_logprobs = torch.log_softmax(self.logits[prompt_ids], dim=-1).gather(1, action_ids[:, None])
        loss, metrics = torch_grpo_loss(
            new_logprobs,
            old_logprobs,
            advantages,
            torch.ones_like(new_logprobs, dtype=torch.bool),
            clip_range=0.2,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.item()), **{key: float(value.item()) for key, value in metrics.items()}}


def run(*, device: str | torch.device | None = None, iterations: int = 100, seed: int = 7) -> ToyRunResult:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(RunConfig(seed=seed, device=str(selected_device)))
    prompt_count, vocab_size, group_size = 16, 8, 8
    correct_tokens = torch.arange(prompt_count, device=selected_device) % vocab_size
    logits = torch.nn.Parameter(torch.zeros(prompt_count, vocab_size, device=selected_device))
    optimizer = torch.optim.AdamW([logits], lr=0.18, weight_decay=0.0)
    initial = _pass_at_1(logits.detach(), correct_tokens)
    final_loss = float("nan")
    final_reward = float("nan")
    final_iteration_seconds = float("nan")
    controller = Controller(
        rollout_worker=ToyRolloutWorker(logits, correct_tokens, group_size),
        reward_worker=RuleRewardWorker(_answer_reward),
        trainer_worker=ToyTrainerWorker(logits, optimizer, selected_device),
    )

    for _ in range(iterations):
        result = controller.run_iteration()
        final_loss = result.metrics["loss"]
        final_reward = result.mean_reward
        final_iteration_seconds = result.timings.iteration_seconds

    final = _pass_at_1(logits.detach(), correct_tokens)
    return ToyRunResult(initial, final, final_reward, final_loss, final_iteration_seconds)


def main() -> None:
    result = run()
    print(f"initial_pass@1={result.initial_pass_at_1:.3f}")
    print(f"final_pass@1={result.final_pass_at_1:.3f}")
    print(f"last_rollout_mean_reward={result.final_reward:.3f}")
    print(f"last_grpo_loss={result.final_loss:.6f}")
    print(f"last_iteration_seconds={result.final_iteration_seconds:.6f}")
    if result.final_pass_at_1 <= result.initial_pass_at_1:
        raise RuntimeError("toy GRPO run did not improve policy pass@1")


if __name__ == "__main__":
    main()
