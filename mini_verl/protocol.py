"""The data contract between rollout, reward, and training workers.

The first version deliberately uses immutable Python objects.  A real backend may
replace these fields with packed tensors, but keeping the semantics explicit here
makes correctness tests independent from PyTorch or a serving engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Iterable, Mapping


class TrajectoryValidationError(ValueError):
    """Raised when a trajectory would be unsafe or ambiguous to train on."""


def _as_int_tuple(name: str, values: Iterable[int], *, allow_empty: bool) -> tuple[int, ...]:
    result = tuple(values)
    if not allow_empty and not result:
        raise TrajectoryValidationError(f"{name} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in result):
        raise TrajectoryValidationError(f"{name} must contain non-negative integer token ids")
    return result


def _as_float_tuple(name: str, values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise TrajectoryValidationError(f"{name} must contain only finite values")
    return result


def _as_finite_optional(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise TrajectoryValidationError(f"{name} must be finite when present")
    return value


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One response sampled by a named policy version.

    `old_logprobs` and `reference_logprobs` are per-response-token values.
    `response_mask` makes padding semantics explicit: tokens with a false mask are
    retained for transport but must not contribute to policy or KL losses.
    """

    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    policy_version: int
    group_id: str
    response_mask: tuple[bool, ...] | None = None
    reference_logprobs: tuple[float, ...] | None = None
    reward: float | None = None
    advantage: float | None = None
    prompt_text: str | None = None
    response_text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        prompt = _as_int_tuple("prompt_token_ids", self.prompt_token_ids, allow_empty=False)
        response = _as_int_tuple("response_token_ids", self.response_token_ids, allow_empty=False)
        old_logprobs = _as_float_tuple("old_logprobs", self.old_logprobs)
        if len(old_logprobs) != len(response):
            raise TrajectoryValidationError(
                "old_logprobs must have exactly one value per response token"
            )
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int) or self.policy_version < 0:
            raise TrajectoryValidationError("policy_version must be a non-negative integer")
        if not isinstance(self.group_id, str) or not self.group_id:
            raise TrajectoryValidationError("group_id must be a non-empty string")

        mask = tuple(True for _ in response) if self.response_mask is None else tuple(self.response_mask)
        if len(mask) != len(response) or any(not isinstance(value, bool) for value in mask):
            raise TrajectoryValidationError(
                "response_mask must contain one boolean per response token"
            )
        if not any(mask):
            raise TrajectoryValidationError("response_mask must include at least one trainable token")

        reference = self.reference_logprobs
        if reference is not None:
            reference = _as_float_tuple("reference_logprobs", reference)
            if len(reference) != len(response):
                raise TrajectoryValidationError(
                    "reference_logprobs must have exactly one value per response token"
                )

        if self.prompt_text is not None and not isinstance(self.prompt_text, str):
            raise TrajectoryValidationError("prompt_text must be a string when present")
        if self.response_text is not None and not isinstance(self.response_text, str):
            raise TrajectoryValidationError("response_text must be a string when present")
        if not isinstance(self.metadata, Mapping):
            raise TrajectoryValidationError("metadata must be a mapping")

        object.__setattr__(self, "prompt_token_ids", prompt)
        object.__setattr__(self, "response_token_ids", response)
        object.__setattr__(self, "old_logprobs", old_logprobs)
        object.__setattr__(self, "response_mask", mask)
        object.__setattr__(self, "reference_logprobs", reference)
        object.__setattr__(self, "reward", _as_finite_optional("reward", self.reward))
        object.__setattr__(self, "advantage", _as_finite_optional("advantage", self.advantage))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def response_token_count(self) -> int:
        """Number of unmasked tokens contributing to training."""
        return sum(self.response_mask)

    def with_reward(self, reward: float) -> "Trajectory":
        return replace(self, reward=reward)

    def with_advantage(self, advantage: float) -> "Trajectory":
        return replace(self, advantage=advantage)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for future worker transport."""
        return {
            "prompt_token_ids": list(self.prompt_token_ids),
            "response_token_ids": list(self.response_token_ids),
            "old_logprobs": list(self.old_logprobs),
            "policy_version": self.policy_version,
            "group_id": self.group_id,
            "response_mask": list(self.response_mask),
            "reference_logprobs": None if self.reference_logprobs is None else list(self.reference_logprobs),
            "reward": self.reward,
            "advantage": self.advantage,
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Trajectory":
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    """An immutable batch transported between RL framework stages."""

    trajectories: tuple[Trajectory, ...]

    def __post_init__(self) -> None:
        trajectories = tuple(self.trajectories)
        if not trajectories:
            raise TrajectoryValidationError("TrajectoryBatch must not be empty")
        if any(not isinstance(trajectory, Trajectory) for trajectory in trajectories):
            raise TrajectoryValidationError("TrajectoryBatch can only contain Trajectory objects")
        object.__setattr__(self, "trajectories", trajectories)

    @classmethod
    def from_iterable(cls, trajectories: Iterable[Trajectory]) -> "TrajectoryBatch":
        return cls(tuple(trajectories))

    @property
    def response_token_count(self) -> int:
        return sum(trajectory.response_token_count for trajectory in self.trajectories)

    @property
    def policy_versions(self) -> frozenset[int]:
        return frozenset(trajectory.policy_version for trajectory in self.trajectories)

    def require_policy_version(self, expected: int) -> None:
        unexpected = self.policy_versions - {expected}
        if unexpected:
            raise TrajectoryValidationError(
                f"batch contains stale or unexpected policy versions: {sorted(unexpected)}; expected {expected}"
            )

    def groups(self) -> dict[str, tuple[Trajectory, ...]]:
        grouped: dict[str, list[Trajectory]] = {}
        for trajectory in self.trajectories:
            grouped.setdefault(trajectory.group_id, []).append(trajectory)
        return {group_id: tuple(group) for group_id, group in grouped.items()}

    def to_dict(self) -> dict[str, Any]:
        return {"trajectories": [trajectory.to_dict() for trajectory in self.trajectories]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrajectoryBatch":
        return cls.from_iterable(Trajectory.from_dict(item) for item in payload["trajectories"])
