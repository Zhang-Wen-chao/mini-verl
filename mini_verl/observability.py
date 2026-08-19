"""Small, explicit runtime measurements used by examples and benchmarks.

The module intentionally reports only values that PyTorch directly exposes.  In
particular, CUDA allocator peak memory is not GPU utilization; utilization needs
an external sampler such as NVML over a defined sampling interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from time import monotonic
from typing import Any
import subprocess


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("CUDA memory measurement requires PyTorch") from error
    return torch


@dataclass(frozen=True, slots=True)
class CudaMemoryStats:
    """PyTorch allocator values for one measured CUDA interval, in bytes."""

    max_allocated_bytes: int
    max_reserved_bytes: int

    def as_metrics(self) -> dict[str, float]:
        return {
            "cuda_max_allocated_bytes": float(self.max_allocated_bytes),
            "cuda_max_reserved_bytes": float(self.max_reserved_bytes),
        }


@dataclass(slots=True)
class CudaMemoryMonitor:
    """Measure peak PyTorch allocator usage during an interval.

    CPU devices are deliberately a no-op so the same benchmark can be executed
    in a local dependency-light environment.  CUDA operations are synchronized
    before resetting and reading peaks, ensuring completed work belongs to the
    intended interval.
    """

    device: str
    _started: bool = False
    _uses_cuda: bool = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("CUDA memory monitor is already running")
        if self.device == "cpu":
            self._uses_cuda = False
            self._started = True
            return
        torch = _torch()
        resolved = torch.device(self.device)
        if resolved.type != "cuda":
            self._uses_cuda = False
            self._started = True
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA memory measurement requested but CUDA is unavailable")
        torch.cuda.synchronize(resolved)
        torch.cuda.reset_peak_memory_stats(resolved)
        self._uses_cuda = True
        self._started = True

    def stop(self) -> CudaMemoryStats | None:
        if not self._started:
            raise RuntimeError("CUDA memory monitor was not started")
        self._started = False
        if not self._uses_cuda:
            return None
        torch = _torch()
        resolved = torch.device(self.device)
        torch.cuda.synchronize(resolved)
        return CudaMemoryStats(
            max_allocated_bytes=int(torch.cuda.max_memory_allocated(resolved)),
            max_reserved_bytes=int(torch.cuda.max_memory_reserved(resolved)),
        )


@dataclass(frozen=True, slots=True)
class GpuUtilizationStats:
    """Device-level GPU samples collected over one benchmark interval.

    These values include every process on the physical GPU.  They are useful
    only when the selected device is known to be idle or exclusively owned.
    """

    sample_count: int
    mean_utilization_percent: float
    max_utilization_percent: int
    max_memory_used_mib: int

    def as_metrics(self) -> dict[str, float]:
        return {
            "gpu_utilization_sample_count": float(self.sample_count),
            "gpu_utilization_mean_percent": self.mean_utilization_percent,
            "gpu_utilization_max_percent": float(self.max_utilization_percent),
            "gpu_memory_used_max_mib": float(self.max_memory_used_mib),
        }


def _parse_nvidia_smi_sample(output: str) -> tuple[int, int]:
    """Parse one `utilization.gpu,memory.used` no-header nvidia-smi row."""
    fields = [field.strip() for field in output.strip().split(",")]
    if len(fields) != 2:
        raise RuntimeError(f"unexpected nvidia-smi output: {output!r}")
    try:
        utilization, memory_used = (int(field) for field in fields)
    except ValueError as error:
        raise RuntimeError(f"nvidia-smi returned non-numeric metrics: {output!r}") from error
    if not 0 <= utilization <= 100 or memory_used < 0:
        raise RuntimeError(f"nvidia-smi returned out-of-range metrics: {output!r}")
    return utilization, memory_used


class GpuUtilizationMonitor:
    """Poll `nvidia-smi` at a fixed interval without adding a Python NVML dependency.

    Sampling has intentional operational constraints: `gpu_index` is a physical
    nvidia-smi index, which may differ from a CUDA logical index after
    `CUDA_VISIBLE_DEVICES` remapping.  The monitor gathers device-level rather
    than process-level metrics and therefore must only be interpreted on a known
    idle/exclusive GPU.
    """

    def __init__(self, gpu_index: int, *, interval_seconds: float = 0.1) -> None:
        if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
            raise ValueError("gpu_index must be a non-negative integer")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.gpu_index = gpu_index
        self.interval_seconds = interval_seconds
        self._samples: list[tuple[int, int]] = []
        self._stop_event = Event()
        self._thread: Thread | None = None

    def _query_once(self) -> tuple[int, int]:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=max(1.0, self.interval_seconds * 4),
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("nvidia-smi utilization sampling failed") from error
        return _parse_nvidia_smi_sample(completed.stdout)

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._samples.append(self._query_once())
            except RuntimeError:
                # The caller receives the empty/partial result rather than a
                # background-thread traceback.  Command availability is checked
                # synchronously in start().
                return
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU utilization monitor is already running")
        # Fail at the call site if the binary/device is invalid. The first sample
        # also includes the beginning of the measured interval.
        self._samples = [self._query_once()]
        self._stop_event.clear()
        self._thread = Thread(target=self._poll, name="mini-verl-gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> GpuUtilizationStats:
        if self._thread is None:
            raise RuntimeError("GPU utilization monitor was not started")
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        if not self._samples:
            raise RuntimeError("GPU utilization monitor collected no samples")
        utilizations, memory_used = zip(*self._samples, strict=True)
        return GpuUtilizationStats(
            sample_count=len(self._samples),
            mean_utilization_percent=sum(utilizations) / len(utilizations),
            max_utilization_percent=max(utilizations),
            max_memory_used_mib=max(memory_used),
        )
