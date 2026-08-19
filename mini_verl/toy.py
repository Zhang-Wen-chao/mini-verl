"""The smallest complete GRPO training loop in :mod:`mini_verl`.

This module intentionally replaces a language model with a categorical policy:
each prompt has one correct token, a rollout samples several candidate tokens,
and the normal controller applies rule reward, group-relative advantages and the
clipped GRPO update.  It keeps the important data-flow and policy-version
semantics visible without a model download, tokenizer or GPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .algorithms.grpo import torch_grpo_loss
from .config import RunConfig, seed_everything
from .controller import Controller
from .protocol import Trajectory, TrajectoryBatch
from .workers import RuleRewardWorker


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("the mini GRPO example requires PyTorch; install mini-verl[torch]") from error
    return torch


@dataclass(frozen=True, slots=True)
class ToyRunResult:
    """The small set of values needed to verify the training loop learned."""

    initial_pass_at_1: float
    final_pass_at_1: float
    final_reward: float
    final_loss: float
    final_iteration_seconds: float
    completed_iterations: int


def _answer_reward(trajectory: Trajectory) -> float:
    return float(trajectory.response_token_ids[0] == trajectory.metadata["correct_token"])


def _sample_batch(
    logits: Any,
    correct_tokens: Any,
    *,
    group_size: int,
    policy_version: int,
) -> TrajectoryBatch:
    torch = _torch()
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


def _pass_at_1(logits: Any, correct_tokens: Any) -> float:
    return float((logits.argmax(dim=-1) == correct_tokens).float().mean().item())


@dataclass(slots=True)
class ToyRolloutWorker:
    """Generate G one-token trajectories for every categorical prompt."""

    logits: Any
    correct_tokens: Any
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
    """Apply the same masked GRPO loss as a CausalLM backend, on one token."""

    logits: Any
    optimizer: Any
    device: Any

    def train(self, batch: TrajectoryBatch, *, learner_policy_version: int) -> Mapping[str, float]:
        torch = _torch()
        if learner_policy_version < 0:
            raise ValueError("learner_policy_version must be non-negative")
        if len(batch.policy_versions) != 1:
            raise ValueError("trainer requires one rollout policy version per batch")
        if any(trajectory.advantage is None for trajectory in batch.trajectories):
            raise ValueError("trainer requires group-relative advantages")

        prompt_ids = torch.tensor([t.prompt_token_ids[0] for t in batch.trajectories], device=self.device)
        action_ids = torch.tensor([t.response_token_ids[0] for t in batch.trajectories], device=self.device)
        old_logprobs = torch.tensor([[t.old_logprobs[0]] for t in batch.trajectories], device=self.device)
        advantages = torch.tensor([float(t.advantage) for t in batch.trajectories], device=self.device)
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


def run_toy_grpo(
    *,
    device: str | None = None,
    iterations: int = 100,
    seed: int = 7,
    prompt_count: int = 16,
    vocab_size: int = 8,
    group_size: int = 8,
    learning_rate: float = 0.18,
) -> ToyRunResult:
    """Train the categorical policy and return an auditable learning result.

    ``group_size`` must be at least two: GRPO normalizes reward within the
    prompt group, so a one-sample group cannot produce a learning signal.
    """
    if iterations <= 0 or prompt_count <= 0 or vocab_size <= 1:
        raise ValueError("iterations and prompt_count must be positive; vocab_size must exceed one")
    if group_size < 2:
        raise ValueError("group_size must be at least two for group-relative GRPO")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    torch = _torch()
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(RunConfig(seed=seed, device=str(selected_device)))
    correct_tokens = torch.arange(prompt_count, device=selected_device) % vocab_size
    logits = torch.nn.Parameter(torch.zeros(prompt_count, vocab_size, device=selected_device))
    optimizer = torch.optim.AdamW([logits], lr=learning_rate, weight_decay=0.0)
    controller = Controller(
        rollout_worker=ToyRolloutWorker(logits, correct_tokens, group_size),
        reward_worker=RuleRewardWorker(_answer_reward),
        trainer_worker=ToyTrainerWorker(logits, optimizer, selected_device),
    )
    initial = _pass_at_1(logits.detach(), correct_tokens)
    result = None
    for _ in range(iterations):
        result = controller.run_iteration()

    assert result is not None
    return ToyRunResult(
        initial_pass_at_1=initial,
        final_pass_at_1=_pass_at_1(logits.detach(), correct_tokens),
        final_reward=result.mean_reward,
        final_loss=result.metrics["loss"],
        final_iteration_seconds=result.timings.iteration_seconds,
        completed_iterations=controller.policy_version,
    )


# Compatibility alias for the original example import. New callers should use
# the descriptive public name above.
run = run_toy_grpo


def main() -> None:
    """Run the minimal GRPO loop as ``python -m mini_verl.toy``."""
    result = run_toy_grpo()
    print(f"initial_pass@1={result.initial_pass_at_1:.3f}")
    print(f"final_pass@1={result.final_pass_at_1:.3f}")
    print(f"last_rollout_mean_reward={result.final_reward:.3f}")
    print(f"last_grpo_loss={result.final_loss:.6f}")
    print(f"last_iteration_seconds={result.final_iteration_seconds:.6f}")
    if result.final_pass_at_1 <= result.initial_pass_at_1:
        raise RuntimeError("toy GRPO run did not improve policy pass@1")


if __name__ == "__main__":
    main()
