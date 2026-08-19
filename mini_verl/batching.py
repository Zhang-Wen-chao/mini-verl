"""Length-aware trajectory batching for variable-length LLM RL workloads."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .protocol import Trajectory, TrajectoryBatch


@dataclass(frozen=True, slots=True)
class PackedTrajectoryBatch:
    """A length-aware batch plus accounting used to compare packing strategies."""

    batch: TrajectoryBatch
    padded_sequence_tokens: int
    real_sequence_tokens: int

    @property
    def padding_tokens(self) -> int:
        return self.padded_sequence_tokens - self.real_sequence_tokens

    @property
    def padding_ratio(self) -> float:
        if self.padded_sequence_tokens == 0:
            return 0.0
        return self.padding_tokens / self.padded_sequence_tokens


def sequence_length(trajectory: Trajectory) -> int:
    return len(trajectory.prompt_token_ids) + len(trajectory.response_token_ids)


def length_bucket_batches(
    trajectories: Iterable[Trajectory],
    *,
    max_batch_size: int,
    max_padded_tokens: int,
) -> tuple[PackedTrajectoryBatch, ...]:
    """Greedily batch longest trajectories first under a padded-token budget.

    A batch with `n` trajectories and longest sequence length `L` costs `n * L`
    tokens after right padding. Sorting by descending length keeps similarly sized
    sequences together and makes this simple policy deterministic.
    """
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if max_padded_tokens <= 0:
        raise ValueError("max_padded_tokens must be positive")

    ordered = sorted(tuple(trajectories), key=sequence_length, reverse=True)
    output: list[PackedTrajectoryBatch] = []
    current: list[Trajectory] = []
    current_longest = 0

    def flush() -> None:
        nonlocal current, current_longest
        if not current:
            return
        packed = current_longest * len(current)
        real = sum(sequence_length(trajectory) for trajectory in current)
        output.append(
            PackedTrajectoryBatch(
                batch=TrajectoryBatch.from_iterable(current),
                padded_sequence_tokens=packed,
                real_sequence_tokens=real,
            )
        )
        current = []
        current_longest = 0

    for trajectory in ordered:
        length = sequence_length(trajectory)
        if length > max_padded_tokens:
            raise ValueError(
                f"trajectory sequence length {length} exceeds max_padded_tokens {max_padded_tokens}"
            )
        proposed_size = len(current) + 1
        proposed_longest = max(current_longest, length)
        if current and (proposed_size > max_batch_size or proposed_size * proposed_longest > max_padded_tokens):
            flush()
        current.append(trajectory)
        current_longest = max(current_longest, length)
    flush()
    return tuple(output)
