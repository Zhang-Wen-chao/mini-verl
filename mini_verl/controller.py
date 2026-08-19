"""The synchronous policy-versioned RL iteration controller."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .metrics import StageTimer
from .protocol import TrajectoryValidationError
from .workers import (
    IterationResult,
    PolicySynchronizer,
    RolloutWorker,
    RuleRewardWorker,
    TrainerWorker,
    mean_reward,
)


@dataclass(slots=True)
class Controller:
    """Run `rollout -> reward -> train -> version advance` safely.

    This synchronous controller makes the first correctness invariant explicit:
    training only consumes trajectories sampled from the policy version it claims
    to optimize. Asynchronous replay or policy-lag policies can extend this after
    their semantics are specified and tested.
    """

    rollout_worker: RolloutWorker
    reward_worker: RuleRewardWorker
    trainer_worker: TrainerWorker
    policy_version: int = 0
    policy_synchronizer: PolicySynchronizer | None = None
    stage_synchronizer: Callable[[], None] | None = None

    def run_iteration(self) -> IterationResult:
        version = self.policy_version
        timer = StageTimer(synchronize=self.stage_synchronizer)
        with timer.measure("rollout"):
            rollout = self.rollout_worker.rollout(policy_version=version)
        try:
            rollout.require_policy_version(version)
        except TrajectoryValidationError as error:
            raise TrajectoryValidationError(
                f"rollout worker returned a batch incompatible with policy version {version}"
            ) from error

        with timer.measure("reward"):
            scored = self.reward_worker.score(rollout)
        with timer.measure("train"):
            metrics = {
                key: float(value)
                for key, value in self.trainer_worker.train(scored, learner_policy_version=version).items()
            }
        next_policy_version = version + 1
        policy_handle = None
        if self.policy_synchronizer is not None:
            with timer.measure("sync"):
                policy_handle = self.policy_synchronizer.synchronize(policy_version=next_policy_version)
            if policy_handle.version != next_policy_version:
                raise TrajectoryValidationError(
                    f"policy synchronizer published version {policy_handle.version}, expected {next_policy_version}"
                )
        timings = timer.finish()
        response_tokens = scored.response_token_count
        average_reward = mean_reward(scored)
        metrics.update(timings.as_metrics())
        metrics.update({
            "mean_reward": average_reward,
            "mean_response_tokens": response_tokens / len(scored.trajectories),
            "rollout_tokens_per_second": response_tokens / timings.rollout_seconds if timings.rollout_seconds else float("inf"),
            "train_tokens_per_second": response_tokens / timings.train_seconds if timings.train_seconds else float("inf"),
        })
        self.policy_version = next_policy_version
        return IterationResult(
            policy_version=version,
            next_policy_version=next_policy_version,
            trajectory_count=len(scored.trajectories),
            response_token_count=response_tokens,
            mean_reward=average_reward,
            metrics=metrics,
            timings=timings,
            policy_handle=policy_handle,
        )
