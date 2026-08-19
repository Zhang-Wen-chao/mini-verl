"""Small worker boundaries used by the RL controller.

The workers intentionally run in-process today. Their input/output contracts are
the same boundaries that later become separate processes, Ray actors, or services.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .metrics import StageTimings
from .policy_sync import PolicyHandle
from .protocol import TrajectoryBatch
from .reward import RewardFn, apply_rewards, group_relative_advantages


class RolloutWorker(Protocol):
    """Generate trajectories using precisely the requested policy version."""

    def rollout(self, *, policy_version: int) -> TrajectoryBatch: ...


class TrainerWorker(Protocol):
    """Update the learner policy from scored trajectories.

    ``learner_policy_version`` identifies the parameters currently being
    optimized.  It deliberately need not equal ``batch.policy_versions``: the
    latter identifies the rollout policy that produced ``old_logprobs``.  This
    distinction is required for bounded policy-lag GRPO.
    """

    def train(self, batch: TrajectoryBatch, *, learner_policy_version: int) -> Mapping[str, float]: ...


class PolicySynchronizer(Protocol):
    """Publish a trainer policy update to an independent rollout replica."""

    def synchronize(self, *, policy_version: int) -> PolicyHandle: ...


@dataclass(slots=True)
class RuleRewardWorker:
    """Apply terminal rewards and GRPO group-relative normalization."""

    reward_fn: RewardFn
    epsilon: float = 1e-6

    def score(self, batch: TrajectoryBatch) -> TrajectoryBatch:
        return group_relative_advantages(apply_rewards(batch, self.reward_fn), epsilon=self.epsilon)


@dataclass(frozen=True, slots=True)
class IterationResult:
    """Immutable audit record for one completed controller iteration."""

    policy_version: int
    next_policy_version: int
    trajectory_count: int
    response_token_count: int
    mean_reward: float
    metrics: Mapping[str, float]
    timings: StageTimings
    policy_handle: PolicyHandle | None = None


def mean_reward(batch: TrajectoryBatch) -> float:
    rewards = [trajectory.reward for trajectory in batch.trajectories]
    if any(reward is None for reward in rewards):
        raise ValueError("mean_reward requires scored trajectories")
    return sum(float(reward) for reward in rewards) / len(rewards)
