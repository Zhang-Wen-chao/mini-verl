"""Measure length bucketing with the real Hugging Face GRPO trainer path.

The benchmark uses the exact same 32 synthetic trajectories, model, number of
optimizer steps, and sequence-length distribution in both modes.  Only the
composition of each batch differs:

* ``mixed`` alternates short and long responses within every batch;
* ``bucketed`` uses :func:`length_bucket_batches` to group similar lengths.

It therefore attributes differences in padded-token accounting and train wall
time to batching rather than to a changed dataset or number of optimizer steps.
The local tiny GPT-2 is an integration/regression workload, not a production
model performance claim.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from time import perf_counter

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from mini_verl.batching import PackedTrajectoryBatch, length_bucket_batches, sequence_length
from mini_verl.config import RunConfig, seed_everything
from mini_verl.hf import HuggingFaceTrainerWorker
from mini_verl.observability import CudaMemoryMonitor, GpuUtilizationMonitor
from mini_verl.protocol import Trajectory, TrajectoryBatch


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    strategy: str
    device: str
    epochs: int
    warmup: int
    repeats: int
    batch_size: int
    sample_gpu_utilization: bool
    gpu_index: int

    def __post_init__(self) -> None:
        if self.strategy not in {"mixed", "bucketed"}:
            raise ValueError("strategy must be 'mixed' or 'bucketed'")
        if self.epochs <= 0 or self.warmup < 0 or self.repeats <= 0 or self.batch_size <= 0:
            raise ValueError("epochs/repeats/batch_size must be positive and warmup must be non-negative")
        if self.batch_size != 4:
            raise ValueError("this controlled workload fixes batch_size=4 to preserve equal optimizer steps")


def trajectories() -> tuple[Trajectory, ...]:
    """Return 32 fixed trajectories with four deliberately different lengths."""
    output: list[Trajectory] = []
    for response_length in (2, 4, 8, 16):
        for copy_index in range(8):
            index = len(output)
            response = tuple(8 + (index * 3 + token_index) % 56 for token_index in range(response_length))
            output.append(
                Trajectory(
                    prompt_token_ids=(2, 3, 4, 5),
                    response_token_ids=response,
                    old_logprobs=tuple(-4.0 for _ in response),
                    policy_version=0,
                    group_id=f"group-{index // 2}",
                    advantage=1.0 if index % 2 else -1.0,
                )
            )
    return tuple(output)


def _packed(batch: TrajectoryBatch) -> PackedTrajectoryBatch:
    longest = max(sequence_length(trajectory) for trajectory in batch.trajectories)
    return PackedTrajectoryBatch(
        batch=batch,
        padded_sequence_tokens=longest * len(batch.trajectories),
        real_sequence_tokens=sum(sequence_length(trajectory) for trajectory in batch.trajectories),
    )


def batches(strategy: str) -> tuple[PackedTrajectoryBatch, ...]:
    items = trajectories()
    if strategy == "bucketed":
        return length_bucket_batches(items, max_batch_size=4, max_padded_tokens=80)
    by_length = [items[offset : offset + 8] for offset in range(0, len(items), 8)]
    mixed = tuple(item for row in zip(*by_length, strict=True) for item in row)
    return tuple(
        _packed(TrajectoryBatch.from_iterable(mixed[offset : offset + 4]))
        for offset in range(0, len(mixed), 4)
    )


def _model(device: torch.device) -> GPT2LMHeadModel:
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_embd=64,
            n_layer=2,
            n_head=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=1,
        )
    ).to(device)


def run_once(config: BenchmarkConfig, *, seed: int) -> dict[str, float]:
    seed_everything(RunConfig(seed=seed, device=config.device, deterministic=False))
    device = torch.device(config.device)
    packed_batches = batches(config.strategy)
    model = _model(device)
    trainer = HuggingFaceTrainerWorker(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        pad_token_id=0,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    for _ in range(config.epochs):
        for packed in packed_batches:
            trainer.train(packed.batch, learner_policy_version=0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    real_sequence_tokens = sum(packed.real_sequence_tokens for packed in packed_batches) * config.epochs
    padded_sequence_tokens = sum(packed.padded_sequence_tokens for packed in packed_batches) * config.epochs
    response_tokens = sum(packed.batch.response_token_count for packed in packed_batches) * config.epochs
    return {
        "train_seconds": elapsed,
        "optimizer_steps": float(len(packed_batches) * config.epochs),
        "real_sequence_tokens": float(real_sequence_tokens),
        "padded_sequence_tokens": float(padded_sequence_tokens),
        "response_tokens": float(response_tokens),
        "padding_ratio": (padded_sequence_tokens - real_sequence_tokens) / padded_sequence_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=("mixed", "bucketed"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-gpu-utilization", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    config = BenchmarkConfig(
        strategy=args.strategy,
        device=args.device,
        epochs=args.epochs,
        warmup=args.warmup,
        repeats=args.repeats,
        batch_size=args.batch_size,
        sample_gpu_utilization=args.sample_gpu_utilization,
        gpu_index=args.gpu_index,
    )
    for repeat in range(config.warmup):
        run_once(config, seed=10_000 + repeat)

    measurements: list[dict[str, float]] = []
    allocated_peaks: list[int] = []
    reserved_peaks: list[int] = []
    utilization_means: list[float] = []
    utilization_maxima: list[int] = []
    utilization_memory_maxima: list[int] = []
    utilization_sample_counts: list[int] = []
    for repeat in range(config.repeats):
        memory_monitor = CudaMemoryMonitor(config.device)
        utilization_monitor = GpuUtilizationMonitor(config.gpu_index) if config.sample_gpu_utilization else None
        memory_monitor.start()
        if utilization_monitor is not None:
            utilization_monitor.start()
        measurements.append(run_once(config, seed=20_000 + repeat))
        memory = memory_monitor.stop()
        utilization = utilization_monitor.stop() if utilization_monitor is not None else None
        if memory is not None:
            allocated_peaks.append(memory.max_allocated_bytes)
            reserved_peaks.append(memory.max_reserved_bytes)
        if utilization is not None:
            utilization_means.append(utilization.mean_utilization_percent)
            utilization_maxima.append(utilization.max_utilization_percent)
            utilization_memory_maxima.append(utilization.max_memory_used_mib)
            utilization_sample_counts.append(utilization.sample_count)

    report: dict[str, object] = {
        "benchmark": "tiny_hf_grpo_length_bucketing",
        "strategy": config.strategy,
        "device": config.device,
        "model": {"architecture": "GPT2LMHeadModel", "vocab_size": 64, "n_embd": 64, "n_layer": 2, "n_head": 2},
        "length_distribution": {"prompt_tokens": 4, "response_lengths": [2, 4, 8, 16], "count_per_length": 8},
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "warmup": config.warmup,
        "repeats": config.repeats,
    }
    for key in (
        "train_seconds", "optimizer_steps", "real_sequence_tokens", "padded_sequence_tokens",
        "response_tokens", "padding_ratio",
    ):
        report[f"{key}_median"] = round(statistics.median(run[key] for run in measurements), 6)
    train_seconds = float(report["train_seconds_median"])
    report["real_sequence_tokens_per_second"] = round(
        float(report["real_sequence_tokens_median"]) / train_seconds, 3
    )
    report["padded_sequence_tokens_per_second"] = round(
        float(report["padded_sequence_tokens_median"]) / train_seconds, 3
    )
    report["response_tokens_per_second"] = round(
        float(report["response_tokens_median"]) / train_seconds, 3
    )
    if allocated_peaks:
        report["cuda_max_allocated_bytes_median"] = int(statistics.median(allocated_peaks))
        report["cuda_max_reserved_bytes_median"] = int(statistics.median(reserved_peaks))
    if utilization_means:
        report.update({
            "gpu_utilization_scope": "device_level_nvidia_smi",
            "gpu_physical_index": config.gpu_index,
            "gpu_utilization_interval_seconds": 0.1,
            "gpu_utilization_mean_percent_median": round(statistics.median(utilization_means), 3),
            "gpu_utilization_max_percent_max": max(utilization_maxima),
            "gpu_memory_used_max_mib_max": max(utilization_memory_maxima),
            "gpu_utilization_sample_count_median": int(statistics.median(utilization_sample_counts)),
        })
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
