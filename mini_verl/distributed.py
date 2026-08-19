"""Minimal torch.distributed primitives shared by distributed training backends."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """The process identity and device selected by a torchrun launch."""

    rank: int
    world_size: int
    local_rank: int
    device: Any

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def initialize_distributed(*, backend: str | None = None) -> DistributedContext:
    """Initialize from torchrun RANK/WORLD_SIZE/LOCAL_RANK environment variables."""
    try:
        import torch
        import torch.distributed as dist
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("distributed helpers require PyTorch") from error

    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError as error:
        raise RuntimeError("initialize_distributed must be launched with torchrun") from error
    if world_size < 2:
        raise ValueError("distributed launch requires WORLD_SIZE >= 2")
    if not torch.cuda.is_available():
        raise RuntimeError("the initial distributed backend requires CUDA")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise ValueError(f"LOCAL_RANK {local_rank} is not a visible CUDA device")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend or "nccl")
    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=torch.device(f"cuda:{local_rank}"),
    )


def is_distributed() -> bool:
    try:
        import torch.distributed as dist
    except ModuleNotFoundError:
        return False
    return dist.is_available() and dist.is_initialized()


def mean_across_ranks(value: Any) -> Any:
    """Return a detached scalar mean without changing the caller tensor."""
    if not is_distributed():
        return value.detach()
    import torch.distributed as dist

    result = value.detach().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    result /= dist.get_world_size()
    return result


def destroy_distributed() -> None:
    """Tear down a process group created by a short-lived training process."""
    if is_distributed():
        import torch.distributed as dist

        dist.destroy_process_group()
