"""Compare synchronous and one-step-lag HF GRPO pipelines on two GPUs.

Both modes use independent trainer and rollout replicas: trainer work runs on
``--trainer-device`` and generation runs on ``--rollout-device``.  The only
difference is schedule: synchronous mode does rollout -> train -> sync, while
prefetch mode overlaps next rollout with learner work and allows exactly one
policy version of lag.  The intentionally tiny local GPT-2 makes this a
reproducible scheduling regression test, not a production-throughput result.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from mini_verl.async_controller import PrefetchingController
from mini_verl.config import RunConfig, seed_everything
from mini_verl.controller import Controller
from mini_verl.hf import HuggingFaceRolloutWorker, HuggingFaceTrainerWorker, PromptExample
from mini_verl.observability import CudaMemoryMonitor, GpuUtilizationMonitor
from mini_verl.pipeline import AsyncRolloutBuffer
from mini_verl.policy_sync import ModelPolicySynchronizer, PolicyHandle
from mini_verl.workers import RuleRewardWorker

from tiny_hf_grpo_benchmark import TinyTokenizer


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    pipeline: str
    trainer_device: str
    rollout_device: str
    iterations: int
    warmup: int
    repeats: int
    prompt_count: int
    group_size: int
    max_new_tokens: int
    rollout_batch_size: int
    prompt_lengths: tuple[int, ...] | None
    bucket_prompts_by_length: bool
    rollout_max_padded_prompt_tokens: int | None
    rollout_max_padded_sequence_tokens: int | None
    train_micro_batch_size: int | None
    train_max_padded_tokens: int | None
    sample_gpu_utilization: bool
    trainer_gpu_index: int
    rollout_gpu_index: int

    def __post_init__(self) -> None:
        if self.pipeline not in {"synchronous", "prefetch"}:
            raise ValueError("pipeline must be 'synchronous' or 'prefetch'")
        if self.iterations <= 0 or self.warmup < 0 or self.repeats <= 0:
            raise ValueError("iterations/repeats must be positive and warmup must be non-negative")
        if (
            self.prompt_count <= 0
            or self.group_size <= 0
            or self.max_new_tokens <= 0
            or self.rollout_batch_size <= 0
        ):
            raise ValueError("prompt_count, group_size, max_new_tokens, and rollout_batch_size must be positive")
        if self.prompt_lengths is not None:
            if len(self.prompt_lengths) != self.prompt_count or min(self.prompt_lengths) <= 0:
                raise ValueError("prompt_lengths must contain one positive length per prompt")
        if self.rollout_max_padded_prompt_tokens is not None and self.rollout_max_padded_prompt_tokens <= 0:
            raise ValueError("rollout_max_padded_prompt_tokens must be positive when set")
        if self.rollout_max_padded_sequence_tokens is not None and self.rollout_max_padded_sequence_tokens <= 0:
            raise ValueError("rollout_max_padded_sequence_tokens must be positive when set")
        if self.train_micro_batch_size is not None and self.train_micro_batch_size <= 0:
            raise ValueError("train_micro_batch_size must be positive when set")
        if self.train_max_padded_tokens is not None and self.train_max_padded_tokens <= 0:
            raise ValueError("train_max_padded_tokens must be positive when set")


def _model(device: torch.device, *, n_positions: int) -> GPT2LMHeadModel:
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_positions=n_positions,
            n_embd=64,
            n_layer=2,
            n_head=2,
            pad_token_id=TinyTokenizer.pad_token_id,
            bos_token_id=TinyTokenizer.eos_token_id,
            eos_token_id=TinyTokenizer.eos_token_id,
        )
    ).to(device)


def _synchronize_devices(*devices: torch.device) -> None:
    for device in dict.fromkeys(devices):
        if device.type == "cuda":
            torch.cuda.synchronize(device)


@dataclass(slots=True)
class _SynchronizedTrainer:
    worker: HuggingFaceTrainerWorker
    device: torch.device

    def train(self, batch: Any, *, learner_policy_version: int) -> Mapping[str, float]:
        result = self.worker.train(batch, learner_policy_version=learner_policy_version)
        _synchronize_devices(self.device)
        return result


@dataclass(slots=True)
class _SynchronizedPolicySync:
    synchronizer: ModelPolicySynchronizer
    trainer_device: torch.device
    rollout_device: torch.device

    def synchronize(self, *, policy_version: int) -> PolicyHandle:
        handle = self.synchronizer.synchronize(policy_version=policy_version)
        _synchronize_devices(self.trainer_device, self.rollout_device)
        return handle


def _build(
    config: PipelineConfig, *, seed: int
) -> tuple[Controller | PrefetchingController, AsyncRolloutBuffer | None]:
    seed_everything(RunConfig(seed=seed, device=config.trainer_device, deterministic=False))
    trainer_device = torch.device(config.trainer_device)
    rollout_device = torch.device(config.rollout_device)
    n_positions = max(32, (max(config.prompt_lengths) if config.prompt_lengths else 3) + config.max_new_tokens)
    trainer_model = _model(trainer_device, n_positions=n_positions)
    rollout_model = copy.deepcopy(trainer_model).to(rollout_device)
    prompts = [
        PromptExample(
            f"length={length}" if config.prompt_lengths is not None else f"synthetic prompt {index}",
            {"prompt_index": index, "prompt_length": length}
            if config.prompt_lengths is not None
            else {"prompt_index": index},
        )
        for index, length in enumerate(config.prompt_lengths or (0,) * config.prompt_count)
    ]
    rollout_worker = HuggingFaceRolloutWorker(
        model=rollout_model,
        tokenizer=TinyTokenizer(),
        prompts=prompts,
        group_size=config.group_size,
        max_new_tokens=config.max_new_tokens,
        rollout_batch_size=config.rollout_batch_size,
        bucket_prompts_by_length=config.bucket_prompts_by_length,
        rollout_max_padded_prompt_tokens=config.rollout_max_padded_prompt_tokens,
        rollout_max_padded_sequence_tokens=config.rollout_max_padded_sequence_tokens,
        do_sample=True,
        collect_generation_timings=True,
    )
    trainer_worker = _SynchronizedTrainer(
        HuggingFaceTrainerWorker(
            model=trainer_model,
            optimizer=torch.optim.AdamW(trainer_model.parameters(), lr=1e-3),
            pad_token_id=TinyTokenizer.pad_token_id,
            train_micro_batch_size=config.train_micro_batch_size,
            train_max_padded_tokens=config.train_max_padded_tokens,
        ),
        trainer_device,
    )
    synchronizer = _SynchronizedPolicySync(
        ModelPolicySynchronizer(trainer_model, rollout_model), trainer_device, rollout_device
    )
    reward_worker = RuleRewardWorker(lambda trajectory: float(trajectory.metadata["sample_index"] == 0))
    if config.pipeline == "synchronous":
        return (
            Controller(
                rollout_worker=rollout_worker,
                reward_worker=reward_worker,
                trainer_worker=trainer_worker,
                policy_synchronizer=synchronizer,
            ),
            None,
        )
    buffer = AsyncRolloutBuffer(rollout_worker, max_policy_lag=1)
    return (
        PrefetchingController(
            rollout_buffer=buffer,
            reward_worker=reward_worker,
            trainer_worker=trainer_worker,
            policy_synchronizer=synchronizer,
        ),
        buffer,
    )


def _rollout_worker(controller: Controller | PrefetchingController) -> HuggingFaceRolloutWorker:
    if isinstance(controller, PrefetchingController):
        return controller.rollout_buffer.rollout_worker  # type: ignore[return-value]
    return controller.rollout_worker  # type: ignore[return-value]


def run_once(config: PipelineConfig, *, seed: int) -> dict[str, float]:
    controller, buffer = _build(config, seed=seed)
    context = buffer if buffer is not None else nullcontext()
    stage_values: dict[str, list[float]] = {
        "iteration_wall_seconds": [],
        "rollout_seconds": [],
        "reward_seconds": [],
        "train_seconds": [],
        "sync_seconds": [],
        "response_tokens": [],
        "next_rollout_wall_seconds": [],
        "rollout_wait_seconds": [],
        "prefetch_overlap_seconds": [],
        "prompt_batch_count": [],
        "padded_prompt_tokens": [],
        "max_batch_padded_prompt_tokens": [],
        "max_batch_padded_sequence_tokens": [],
        "prompt_padding_ratio": [],
        "train_microbatch_count": [],
        "train_real_sequence_tokens": [],
        "train_padded_sequence_tokens": [],
        "train_padding_ratio": [],
    }
    with context:
        if isinstance(controller, PrefetchingController):
            controller.prime()
        for _ in range(config.iterations):
            started = perf_counter()
            result = controller.run_iteration()
            _synchronize_devices(torch.device(config.trainer_device), torch.device(config.rollout_device))
            stage_values["iteration_wall_seconds"].append(perf_counter() - started)
            stage_values["rollout_seconds"].append(result.timings.rollout_seconds)
            stage_values["reward_seconds"].append(result.timings.reward_seconds)
            stage_values["train_seconds"].append(result.timings.train_seconds)
            stage_values["sync_seconds"].append(result.timings.sync_seconds)
            stage_values["response_tokens"].append(float(result.response_token_count))
            if isinstance(controller, PrefetchingController):
                stage_values["next_rollout_wall_seconds"].append(
                    result.metrics["next_rollout_wall_seconds"]
                )
                stage_values["rollout_wait_seconds"].append(result.metrics["rollout_wait_seconds"])
                stage_values["prefetch_overlap_seconds"].append(result.metrics["prefetch_overlap_seconds"])
            else:
                stage_values["next_rollout_wall_seconds"].append(result.timings.rollout_seconds)
                stage_values["rollout_wait_seconds"].append(result.timings.rollout_seconds)
                stage_values["prefetch_overlap_seconds"].append(0.0)
            rollout_timings = _rollout_worker(controller).last_rollout_timings
            if rollout_timings is None:
                raise RuntimeError("pipeline benchmark expected rollout stage timings")
            prompt_batching = rollout_timings.prompt_batching
            stage_values["prompt_batch_count"].append(float(prompt_batching.batch_count))
            stage_values["padded_prompt_tokens"].append(float(prompt_batching.padded_prompt_tokens))
            stage_values["max_batch_padded_prompt_tokens"].append(
                float(prompt_batching.max_batch_padded_prompt_tokens)
            )
            stage_values["max_batch_padded_sequence_tokens"].append(
                float(prompt_batching.max_batch_padded_sequence_tokens)
            )
            stage_values["prompt_padding_ratio"].append(prompt_batching.padding_ratio)
            for key in (
                "train_microbatch_count", "train_real_sequence_tokens",
                "train_padded_sequence_tokens", "train_padding_ratio",
            ):
                stage_values[key].append(float(result.metrics[key]))
    return {key: statistics.median(values) for key, values in stage_values.items()}


def _device_monitor(device: str) -> CudaMemoryMonitor:
    return CudaMemoryMonitor(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=("synchronous", "prefetch"), required=True)
    parser.add_argument("--trainer-device", default="cuda:0")
    parser.add_argument("--rollout-device", default="cuda:1")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt-count", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=1)
    parser.add_argument("--prompt-lengths", help="comma-separated synthetic token lengths; count must match --prompt-count")
    parser.add_argument("--bucket-prompts-by-length", action="store_true")
    parser.add_argument("--rollout-max-padded-prompt-tokens", type=int)
    parser.add_argument("--rollout-max-padded-sequence-tokens", type=int)
    parser.add_argument("--train-micro-batch-size", type=int)
    parser.add_argument("--train-max-padded-tokens", type=int)
    parser.add_argument("--sample-gpu-utilization", action="store_true")
    parser.add_argument("--trainer-gpu-index", type=int, default=0)
    parser.add_argument("--rollout-gpu-index", type=int, default=1)
    args = parser.parse_args()
    prompt_lengths = tuple(int(part) for part in args.prompt_lengths.split(",") if part) if args.prompt_lengths else None
    config = PipelineConfig(
        pipeline=args.pipeline,
        trainer_device=args.trainer_device,
        rollout_device=args.rollout_device,
        iterations=args.iterations,
        warmup=args.warmup,
        repeats=args.repeats,
        prompt_count=args.prompt_count,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        rollout_batch_size=args.rollout_batch_size,
        prompt_lengths=prompt_lengths,
        bucket_prompts_by_length=args.bucket_prompts_by_length,
        rollout_max_padded_prompt_tokens=args.rollout_max_padded_prompt_tokens,
        rollout_max_padded_sequence_tokens=args.rollout_max_padded_sequence_tokens,
        train_micro_batch_size=args.train_micro_batch_size,
        train_max_padded_tokens=args.train_max_padded_tokens,
        sample_gpu_utilization=args.sample_gpu_utilization,
        trainer_gpu_index=args.trainer_gpu_index,
        rollout_gpu_index=args.rollout_gpu_index,
    )
    for repeat in range(config.warmup):
        run_once(config, seed=10_000 + repeat)

    measurements: list[dict[str, float]] = []
    trainer_allocated: list[int] = []
    trainer_reserved: list[int] = []
    rollout_allocated: list[int] = []
    rollout_reserved: list[int] = []
    utilization: dict[str, list[float]] = {
        "trainer_mean": [], "trainer_max": [], "trainer_memory": [], "trainer_samples": [],
        "rollout_mean": [], "rollout_max": [], "rollout_memory": [], "rollout_samples": [],
    }
    for repeat in range(config.repeats):
        trainer_memory = _device_monitor(config.trainer_device)
        rollout_memory = _device_monitor(config.rollout_device)
        trainer_util = GpuUtilizationMonitor(config.trainer_gpu_index) if config.sample_gpu_utilization else None
        rollout_util = GpuUtilizationMonitor(config.rollout_gpu_index) if config.sample_gpu_utilization else None
        trainer_memory.start()
        rollout_memory.start()
        if trainer_util is not None:
            trainer_util.start()
            assert rollout_util is not None
            rollout_util.start()
        measurements.append(run_once(config, seed=20_000 + repeat))
        trainer_stats = trainer_memory.stop()
        rollout_stats = rollout_memory.stop()
        if trainer_util is not None:
            assert rollout_util is not None
            for prefix, stats in (("trainer", trainer_util.stop()), ("rollout", rollout_util.stop())):
                utilization[f"{prefix}_mean"].append(stats.mean_utilization_percent)
                utilization[f"{prefix}_max"].append(float(stats.max_utilization_percent))
                utilization[f"{prefix}_memory"].append(float(stats.max_memory_used_mib))
                utilization[f"{prefix}_samples"].append(float(stats.sample_count))
        if trainer_stats is not None:
            trainer_allocated.append(trainer_stats.max_allocated_bytes)
            trainer_reserved.append(trainer_stats.max_reserved_bytes)
        if rollout_stats is not None:
            rollout_allocated.append(rollout_stats.max_allocated_bytes)
            rollout_reserved.append(rollout_stats.max_reserved_bytes)

    report: dict[str, object] = {
        "benchmark": "tiny_hf_grpo_two_gpu_pipeline",
        "pipeline": config.pipeline,
        "trainer_device": config.trainer_device,
        "rollout_device": config.rollout_device,
        "iterations_per_repeat": config.iterations,
        "warmup": config.warmup,
        "repeats": config.repeats,
        "prompt_count": config.prompt_count,
        "group_size": config.group_size,
        "max_new_tokens": config.max_new_tokens,
        "rollout_batch_size": config.rollout_batch_size,
        "prompt_lengths": list(config.prompt_lengths) if config.prompt_lengths is not None else None,
        "bucket_prompts_by_length": config.bucket_prompts_by_length,
        "rollout_max_padded_prompt_tokens": config.rollout_max_padded_prompt_tokens,
        "rollout_max_padded_sequence_tokens": config.rollout_max_padded_sequence_tokens,
        "train_micro_batch_size": config.train_micro_batch_size,
        "train_max_padded_tokens": config.train_max_padded_tokens,
        "model": {"architecture": "GPT2LMHeadModel", "vocab_size": 64, "n_embd": 64, "n_layer": 2, "n_head": 2},
    }
    for key in (
        "iteration_wall_seconds", "rollout_seconds", "reward_seconds", "train_seconds", "sync_seconds",
        "response_tokens", "next_rollout_wall_seconds", "rollout_wait_seconds", "prefetch_overlap_seconds",
        "prompt_batch_count", "padded_prompt_tokens", "max_batch_padded_prompt_tokens",
        "max_batch_padded_sequence_tokens", "prompt_padding_ratio",
        "train_microbatch_count", "train_real_sequence_tokens",
        "train_padded_sequence_tokens", "train_padding_ratio",
    ):
        report[f"{key}_median"] = round(statistics.median(run[key] for run in measurements), 6)
    wall = float(report["iteration_wall_seconds_median"])
    tokens = float(report["response_tokens_median"])
    report["iterations_per_second"] = round(1 / wall, 3) if wall else float("inf")
    report["response_tokens_per_second"] = round(tokens / wall, 3) if wall else float("inf")
    if trainer_allocated:
        report.update({
            "trainer_cuda_max_allocated_bytes_median": int(statistics.median(trainer_allocated)),
            "trainer_cuda_max_reserved_bytes_median": int(statistics.median(trainer_reserved)),
            "rollout_cuda_max_allocated_bytes_median": int(statistics.median(rollout_allocated)),
            "rollout_cuda_max_reserved_bytes_median": int(statistics.median(rollout_reserved)),
        })
    if utilization["trainer_mean"]:
        for prefix, gpu_index in (("trainer", config.trainer_gpu_index), ("rollout", config.rollout_gpu_index)):
            report.update({
                f"{prefix}_gpu_physical_index": gpu_index,
                f"{prefix}_gpu_utilization_mean_percent_median": round(statistics.median(utilization[f"{prefix}_mean"]), 3),
                f"{prefix}_gpu_utilization_max_percent_max": int(max(utilization[f"{prefix}_max"])),
                f"{prefix}_gpu_memory_used_max_mib_max": int(max(utilization[f"{prefix}_memory"])),
                f"{prefix}_gpu_utilization_sample_count_median": int(statistics.median(utilization[f"{prefix}_samples"])),
            })
        report["gpu_utilization_scope"] = "device_level_nvidia_smi"
        report["gpu_utilization_interval_seconds"] = 0.1
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
