"""Explicit, full-weight policy synchronization between trainer and rollout replicas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PolicyHandle:
    """Identity of weights made visible to a rollout worker."""

    version: int
    parameter_tensors: int
    parameter_bytes: int


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("policy synchronization requires PyTorch") from error
    return torch


def synchronize_policy(
    source_model: Any,
    destination_model: Any,
    *,
    policy_version: int,
) -> PolicyHandle:
    """Copy a trainer policy into a rollout replica without aliasing storage.

    This is a correctness-first full state-dict transfer.  It preserves model
    buffers as well as parameters and works across devices through PyTorch's copy
    semantics.  The caller owns when a rollout worker switches handles; versioned
    trajectories can then be checked by Controller before training.
    """
    torch = _torch()
    if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
        raise ValueError("policy_version must be a non-negative integer")

    with torch.no_grad():
        state = source_model.state_dict()
        detached = {name: value.detach().clone() for name, value in state.items()}
        destination_model.load_state_dict(detached, strict=True)
    parameter_bytes = sum(value.numel() * value.element_size() for value in detached.values())
    return PolicyHandle(
        version=policy_version,
        parameter_tensors=len(detached),
        parameter_bytes=parameter_bytes,
    )


@dataclass(slots=True)
class ModelPolicySynchronizer:
    """Controller adapter copying a trainer model to a rollout replica."""

    source_model: Any
    destination_model: Any

    def synchronize(self, *, policy_version: int) -> PolicyHandle:
        return synchronize_policy(
            self.source_model,
            self.destination_model,
            policy_version=policy_version,
        )
