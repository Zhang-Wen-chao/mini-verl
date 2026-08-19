"""Small, dependency-free timing primitives for RL iteration observability."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True, slots=True)
class StageTimings:
    """Wall-clock durations for the synchronous rollout/reward/train pipeline."""

    rollout_seconds: float
    reward_seconds: float
    train_seconds: float
    sync_seconds: float = 0.0

    @property
    def iteration_seconds(self) -> float:
        return self.rollout_seconds + self.reward_seconds + self.train_seconds + self.sync_seconds

    def as_metrics(self) -> dict[str, float]:
        return {
            "rollout_seconds": self.rollout_seconds,
            "reward_seconds": self.reward_seconds,
            "train_seconds": self.train_seconds,
            "sync_seconds": self.sync_seconds,
            "iteration_seconds": self.iteration_seconds,
        }


class StageTimer:
    """A monotonic timer that records one explicit framework stage at a time.

    ``synchronize`` is optional because normal controller execution should not
    force an accelerator-wide barrier.  Benchmarks may provide, for example,
    ``torch.cuda.synchronize`` to turn asynchronous CUDA launches into honest
    stage wall-clock measurements.
    """

    def __init__(self, *, synchronize: Callable[[], None] | None = None) -> None:
        self._durations: dict[str, float] = {}
        self._synchronize = synchronize

    def measure(self, stage: str):
        if stage in self._durations:
            raise ValueError(f"stage {stage!r} has already been measured")
        return _Measurement(self._durations, stage, synchronize=self._synchronize)

    def finish(self) -> StageTimings:
        expected = {"rollout", "reward", "train"}
        missing = expected - self._durations.keys()
        if missing:
            raise ValueError(f"missing stage timings: {sorted(missing)}")
        return StageTimings(
            rollout_seconds=self._durations["rollout"],
            reward_seconds=self._durations["reward"],
            train_seconds=self._durations["train"],
            sync_seconds=self._durations.get("sync", 0.0),
        )


class _Measurement:
    def __init__(
        self, durations: dict[str, float], stage: str, *, synchronize: Callable[[], None] | None
    ) -> None:
        self._durations = durations
        self._stage = stage
        self._synchronize = synchronize
        self._started: float | None = None

    def __enter__(self) -> None:
        if self._synchronize is not None:
            self._synchronize()
        self._started = perf_counter()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        assert self._started is not None
        if self._synchronize is not None:
            self._synchronize()
        self._durations[self._stage] = perf_counter() - self._started
        return False
