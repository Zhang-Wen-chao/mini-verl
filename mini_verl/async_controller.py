"""A one-step-lag prefetch controller for overlapping rollout and learner work.

The synchronous Controller remains the default.  This controller makes one
narrow asynchronous tradeoff explicit: while the learner optimizes batch v_k, an
*independent* rollout replica samples the next batch with v_k.  Once that
generation has finished, the controller copies the updated learner parameters to
the replica as v_(k+1).  The next learner update therefore consumes rollout v_k
under learner v_(k+1), with exactly one bounded policy step of lag.

The rollout worker must own an independent model replica.  Synchronization is
intentionally delayed until the corresponding rollout future has completed, so a
weight copy never mutates a model while its generation thread is using it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import StageTimer
from .pipeline import AsyncRolloutBuffer, BufferedRollout
from .policy_sync import PolicyHandle
from .protocol import TrajectoryValidationError
from .workers import IterationResult, PolicySynchronizer, RuleRewardWorker, TrainerWorker, mean_reward


@dataclass(slots=True)
class PrefetchingController:
    """Overlap rollout(v_k) with learner work and consume it at learner v_(k+1)."""

    rollout_buffer: AsyncRolloutBuffer
    reward_worker: RuleRewardWorker
    trainer_worker: TrainerWorker
    policy_synchronizer: PolicySynchronizer
    policy_version: int = 0
    _current: BufferedRollout | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rollout_buffer.max_policy_lag < 1:
            raise ValueError("PrefetchingController requires AsyncRolloutBuffer(max_policy_lag >= 1)")

    @property
    def is_primed(self) -> bool:
        return self._current is not None

    def prime(self) -> None:
        """Obtain the initial on-policy rollout before beginning overlap."""
        if self._current is not None:
            raise RuntimeError("prefetching controller is already primed")
        self.rollout_buffer.submit(policy_version=self.policy_version)
        self._current = self.rollout_buffer.consume_next(learner_policy_version=self.policy_version)

    def run_iteration(self) -> IterationResult:
        """Train one batch and prefetch one safe, one-step-lag successor."""
        if self._current is None:
            self.prime()
        assert self._current is not None
        current = self._current
        learner_version = self.policy_version
        rollout_version = current.requested_policy_version
        if current.batch.policy_versions != {rollout_version}:
            raise TrajectoryValidationError(
                f"current rollout declared version {rollout_version}, returned {sorted(current.batch.policy_versions)}"
            )
        policy_lag = learner_version - rollout_version
        if policy_lag < 0 or policy_lag > self.rollout_buffer.max_policy_lag:
            raise TrajectoryValidationError(
                f"current rollout policy lag {policy_lag} is outside [0, {self.rollout_buffer.max_policy_lag}]"
            )

        timer = StageTimer()
        # The rollout replica is v_k here.  Generation can overlap reward/train
        # because it is independent of the learner model.
        self.rollout_buffer.submit(policy_version=learner_version)
        with timer.measure("reward"):
            scored = self.reward_worker.score(current.batch)
        with timer.measure("train"):
            metrics = {
                key: float(value)
                for key, value in self.trainer_worker.train(
                    scored, learner_policy_version=learner_version
                ).items()
            }

        # Waiting here measures only the unhidden rollout tail after learner
        # work.  It must complete before synchronizing the rollout replica.
        with timer.measure("rollout"):
            next_current = self.rollout_buffer.consume_next(
                learner_policy_version=learner_version
            )
        next_version = learner_version + 1
        with timer.measure("sync"):
            handle = self.policy_synchronizer.synchronize(policy_version=next_version)
        if handle.version != next_version:
            raise TrajectoryValidationError(
                f"policy synchronizer published version {handle.version}, expected {next_version}"
            )

        timings = timer.finish()
        response_tokens = scored.response_token_count
        average_reward = mean_reward(scored)
        metrics.update(timings.as_metrics())
        metrics.update({
            "mean_reward": average_reward,
            "mean_response_tokens": response_tokens / len(scored.trajectories),
            # ``rollout_seconds`` in StageTimings is the wait tail after train,
            # not full generation.  Throughput must use the completed current
            # batch's end-to-end rollout duration instead.
            "rollout_wall_seconds": current.rollout_wall_seconds,
            "rollout_wait_seconds": timings.rollout_seconds,
            "rollout_tokens_per_second": response_tokens / current.rollout_wall_seconds if current.rollout_wall_seconds else float("inf"),
            "train_tokens_per_second": response_tokens / timings.train_seconds if timings.train_seconds else float("inf"),
            "rollout_policy_version": float(rollout_version),
            "learner_policy_version": float(learner_version),
            "policy_lag": float(policy_lag),
            "next_rollout_policy_lag": float(next_version - next_current.requested_policy_version),
            # These describe the request submitted at the beginning of this
            # iteration and consumed above.  They make pipeline efficiency
            # measurable without pretending the post-train wait is full rollout
            # latency.
            "next_rollout_wall_seconds": next_current.rollout_wall_seconds,
            "prefetch_overlap_seconds": min(
                next_current.rollout_wall_seconds,
                timings.reward_seconds + timings.train_seconds,
            ),
        })
        self._current = next_current
        self.policy_version = next_version
        return IterationResult(
            policy_version=learner_version,
            next_policy_version=next_version,
            trajectory_count=len(scored.trajectories),
            response_token_count=response_tokens,
            mean_reward=average_reward,
            metrics=metrics,
            timings=timings,
            policy_handle=handle,
        )
