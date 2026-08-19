"""Deterministic rewards and group-relative advantages for the first GRPO loop."""

from __future__ import annotations

from collections.abc import Callable
import math
import re

from .protocol import Trajectory, TrajectoryBatch, TrajectoryValidationError

RewardFn = Callable[[Trajectory], float]


def normalize_answer(answer: str) -> str:
    """Normalize a compact exact-answer task without pretending to parse mathematics."""
    return " ".join(answer.strip().casefold().split())


def final_answer(text: str) -> str:
    """Use a final `answer:` line when supplied, otherwise the final non-empty line."""
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        return ""
    matches = re.findall(r"(?:final\s+)?answer\s*:\s*(.+)", text, flags=re.IGNORECASE)
    return matches[-1].strip() if matches else nonempty_lines[-1]


def exact_answer_reward(trajectory: Trajectory) -> float:
    """Return one iff the response's final answer equals metadata['expected_answer']."""
    if trajectory.response_text is None:
        raise TrajectoryValidationError("exact_answer_reward requires trajectory.response_text")
    expected = trajectory.metadata.get("expected_answer")
    if not isinstance(expected, str):
        raise TrajectoryValidationError("exact_answer_reward requires string metadata['expected_answer']")
    return float(normalize_answer(final_answer(trajectory.response_text)) == normalize_answer(expected))


def apply_rewards(batch: TrajectoryBatch, reward_fn: RewardFn) -> TrajectoryBatch:
    """Score every trajectory, rejecting non-finite rewards at the worker boundary."""
    scored: list[Trajectory] = []
    for trajectory in batch.trajectories:
        reward = float(reward_fn(trajectory))
        if not math.isfinite(reward):
            raise TrajectoryValidationError("reward function returned a non-finite value")
        scored.append(trajectory.with_reward(reward))
    return TrajectoryBatch.from_iterable(scored)


def group_relative_advantages(batch: TrajectoryBatch, *, epsilon: float = 1e-6) -> TrajectoryBatch:
    """Attach GRPO advantages using population standard deviation per prompt group.

    A constant-reward group receives zero advantage. This avoids exploding values
    while preserving GRPO's useful reward-shift invariance.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    advantages: dict[int, float] = {}
    for trajectories in batch.groups().values():
        rewards = [trajectory.reward for trajectory in trajectories]
        if any(reward is None for reward in rewards):
            raise TrajectoryValidationError("all trajectories need a reward before advantage computation")
        reward_values = [float(reward) for reward in rewards]
        mean = sum(reward_values) / len(reward_values)
        variance = sum((reward - mean) ** 2 for reward in reward_values) / len(reward_values)
        std = math.sqrt(variance)
        for trajectory, reward in zip(trajectories, reward_values, strict=True):
            advantages[id(trajectory)] = 0.0 if std < epsilon else (reward - mean) / (std + epsilon)

    return TrajectoryBatch.from_iterable(
        trajectory.with_advantage(advantages[id(trajectory)]) for trajectory in batch.trajectories
    )
