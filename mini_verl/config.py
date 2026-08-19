"""Reproducibility configuration shared by examples and benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Minimal run identity recorded alongside a training or benchmark result."""

    seed: int
    device: str
    deterministic: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.device:
            raise ValueError("device must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(config: RunConfig) -> None:
    """Seed Python and, when available, PyTorch CPU/CUDA generators.

    `deterministic=True` requests deterministic PyTorch algorithms where possible.
    It may reject individual nondeterministic operations; callers should keep it
    disabled for throughput benchmarks and enable it for correctness experiments.
    """
    random.seed(config.seed)
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
