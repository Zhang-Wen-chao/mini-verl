"""Atomic, local checkpoints for policy training state.

Checkpoints are trusted local artifacts: they contain optimizer state and are
loaded with PyTorch deserialization. Do not load checkpoint files from untrusted
sources.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """Metadata restored independently of model and optimizer tensors."""

    policy_version: int
    extra: Mapping[str, Any]


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("checkpoint helpers require PyTorch") from error
    return torch


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    policy_version: int,
    extra: Mapping[str, Any] | None = None,
) -> CheckpointState:
    """Atomically persist a complete single-process policy-training state.

    The replacement is atomic on the same filesystem, so an interrupted save
    leaves either the old complete checkpoint or the new complete checkpoint.
    """
    torch = _torch()
    if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
        raise ValueError("policy_version must be a non-negative integer")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_extra = dict(extra or {})
    payload = {
        "format_version": 1,
        "policy_version": policy_version,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": checkpoint_extra,
    }
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return CheckpointState(policy_version=policy_version, extra=checkpoint_extra)


def load_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    map_location: Any = None,
) -> CheckpointState:
    """Restore model, optimizer and RNG state from a trusted local checkpoint."""
    torch = _torch()
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    policy_version = payload.get("policy_version")
    if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
        raise ValueError("checkpoint has invalid policy_version")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.random.set_rng_state(payload["torch_rng_state"])
    cuda_rng_state_all = payload.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state_all)
    extra = payload.get("extra", {})
    if not isinstance(extra, Mapping):
        raise ValueError("checkpoint extra metadata must be a mapping")
    return CheckpointState(policy_version=policy_version, extra=dict(extra))
