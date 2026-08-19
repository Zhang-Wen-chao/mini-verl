"""Bounded asynchronous rollout buffering with explicit policy-lag semantics.

The synchronous Controller is the default correctness path. This module is the
smallest useful step toward train/rollout overlap: rollout jobs may finish while
the learner is updating, but every consumed trajectory is checked against a
declared maximum policy lag. It intentionally does not claim thread safety for a
particular model backend; use independent rollout worker replicas when issuing
more than one concurrent request.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from .protocol import TrajectoryBatch, TrajectoryValidationError
from .workers import RolloutWorker


@dataclass(frozen=True, slots=True)
class BufferedRollout:
    """A completed rollout plus the policy version requested from its worker."""

    requested_policy_version: int
    batch: TrajectoryBatch
    rollout_wall_seconds: float


@dataclass(slots=True)
class AsyncRolloutBuffer:
    """A bounded queue of concurrent rollout requests.

    `max_policy_lag=0` preserves on-policy semantics. A value of one permits a
    rollout sampled under policy v_k to be consumed after the learner advanced to
    v_(k+1). The buffer only owns worker futures; reward and training stay in the
    caller so their resource placement remains explicit.
    """

    rollout_worker: RolloutWorker
    max_inflight: int = 1
    max_policy_lag: int = 0
    _executor: ThreadPoolExecutor = field(init=False, repr=False)
    _futures: list[tuple[int, float, Future[TrajectoryBatch]]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        if self.max_policy_lag < 0:
            raise ValueError("max_policy_lag must be non-negative")
        self._executor = ThreadPoolExecutor(max_workers=self.max_inflight, thread_name_prefix="mini-verl-rollout")

    @property
    def pending_count(self) -> int:
        return len(self._futures)

    def submit(self, *, policy_version: int) -> None:
        """Schedule a rollout request under an exact policy-version contract."""
        if policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if self.pending_count >= self.max_inflight:
            raise RuntimeError("rollout buffer is full; consume a rollout before submitting another")
        submitted_at = perf_counter()
        future = self._executor.submit(self.rollout_worker.rollout, policy_version=policy_version)
        self._futures.append((policy_version, submitted_at, future))

    def consume_next(self, *, learner_policy_version: int) -> BufferedRollout:
        """Wait for FIFO completion and reject future/stale trajectory versions."""
        if learner_policy_version < 0:
            raise ValueError("learner_policy_version must be non-negative")
        if not self._futures:
            raise RuntimeError("rollout buffer is empty")
        requested_version, submitted_at, future = self._futures.pop(0)
        batch = future.result()
        rollout_wall_seconds = perf_counter() - submitted_at
        if batch.policy_versions != {requested_version}:
            raise TrajectoryValidationError(
                f"rollout declared request version {requested_version}, returned {sorted(batch.policy_versions)}"
            )
        lag = learner_policy_version - requested_version
        if lag < 0:
            raise TrajectoryValidationError(
                f"rollout policy version {requested_version} is newer than learner version {learner_policy_version}"
            )
        if lag > self.max_policy_lag:
            raise TrajectoryValidationError(
                f"rollout policy lag {lag} exceeds configured maximum {self.max_policy_lag}"
            )
        return BufferedRollout(
            requested_policy_version=requested_version,
            batch=batch,
            rollout_wall_seconds=rollout_wall_seconds,
        )

    def close(self, *, cancel_pending: bool = True) -> None:
        """Release worker threads. The buffer cannot be reused after close."""
        self._executor.shutdown(wait=True, cancel_futures=cancel_pending)
        self._futures.clear()

    def __enter__(self) -> "AsyncRolloutBuffer":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.close()
        return False
