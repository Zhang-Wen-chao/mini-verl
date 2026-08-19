"""Benchmark the real Hugging Face rollout -> GRPO update code path.

The model is intentionally constructed locally rather than downloaded: this is a
reproducible framework-regression workload, not a language-model quality or
production-throughput claim.  It exercises `generate`, old-logprob recompute,
trajectory packing, reward/advantage calculation, and a causal-LM GRPO update.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from mini_verl.config import RunConfig, seed_everything
from mini_verl.controller import Controller
from mini_verl.hf import HuggingFaceRolloutWorker, HuggingFaceTrainerWorker, PromptExample
from mini_verl.observability import CudaMemoryMonitor, GpuUtilizationMonitor
from mini_verl.workers import RuleRewardWorker


class TinyTokenizer:
    """Offline tokenizer sufficient for a fixed synthetic benchmark corpus."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, *, return_tensors: str, add_special_tokens: bool):
        del return_tensors, add_special_tokens
        if text.startswith("length="):
            length = int(text.removeprefix("length="))
            if length <= 0:
                raise ValueError("synthetic prompt length must be positive")
            token_ids = [2 + index % 62 for index in range(length)]
            return {
                "input_ids": torch.tensor([token_ids], dtype=torch.long),
                "attention_mask": torch.ones((1, length), dtype=torch.long),
            }
        token = 2 + sum(text.encode("utf-8")) % 61
        return {
            "input_ids": torch.tensor([[2, token, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return " ".join(str(int(token)) for token in token_ids)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    device: str
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
    gpu_index: int

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.warmup < 0 or self.repeats <= 0:
            raise ValueError("iterations/repeats must be positive and warmup must be non-negative")
        if self.prompt_count <= 0 or self.group_size <= 0 or self.max_new_tokens <= 0 or self.rollout_batch_size <= 0:
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


def build_controller(config: BenchmarkConfig, *, seed: int) -> Controller:
    seed_everything(RunConfig(seed=seed, device=config.device, deterministic=False))
    device = torch.device(config.device)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_positions=max(32, (max(config.prompt_lengths) if config.prompt_lengths else 3) + config.max_new_tokens),
            n_embd=64,
            n_layer=2,
            n_head=2,
            pad_token_id=TinyTokenizer.pad_token_id,
            bos_token_id=TinyTokenizer.eos_token_id,
            eos_token_id=TinyTokenizer.eos_token_id,
        )
    ).to(device)
    prompts = [
        PromptExample(
            f"length={length}" if config.prompt_lengths is not None else f"synthetic prompt {index}",
            {"prompt_index": index, "prompt_length": length} if config.prompt_lengths is not None else {"prompt_index": index},
        )
        for index, length in enumerate(config.prompt_lengths or (0,) * config.prompt_count)
    ]
    stage_synchronizer = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
    return Controller(
        rollout_worker=HuggingFaceRolloutWorker(
            model=model,
            tokenizer=TinyTokenizer(),
            prompts=prompts,
            group_size=config.group_size,
            max_new_tokens=config.max_new_tokens,
            rollout_batch_size=config.rollout_batch_size,
            bucket_prompts_by_length=config.bucket_prompts_by_length,
            rollout_max_padded_prompt_tokens=config.rollout_max_padded_prompt_tokens,
            rollout_max_padded_sequence_tokens=config.rollout_max_padded_sequence_tokens,
            # GRPO needs multiple samples per prompt.  Transformers rejects
            # greedy generation with num_return_sequences > 1; a seeded sample
            # path is both valid and closer to the intended rollout semantics.
            do_sample=True,
            collect_generation_timings=True,
        ),
        reward_worker=RuleRewardWorker(lambda trajectory: float(trajectory.metadata["sample_index"] == 0)),
        trainer_worker=HuggingFaceTrainerWorker(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            pad_token_id=TinyTokenizer.pad_token_id,
            train_micro_batch_size=config.train_micro_batch_size,
            train_max_padded_tokens=config.train_max_padded_tokens,
        ),
        stage_synchronizer=stage_synchronizer,
    )


def run_once(config: BenchmarkConfig, *, seed: int) -> dict[str, float]:
    controller = build_controller(config, seed=seed)
    stage_values: dict[str, list[float]] = {
        "rollout_seconds": [],
        "reward_seconds": [],
        "train_seconds": [],
        "iteration_seconds": [],
        "response_tokens": [],
        "prefill_seconds": [],
        "decode_seconds": [],
        "prefill_forward_calls": [],
        "decode_forward_calls": [],
        "old_logprob_forward_seconds": [],
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
    for _ in range(config.iterations):
        result = controller.run_iteration()
        stage_values["rollout_seconds"].append(result.timings.rollout_seconds)
        stage_values["reward_seconds"].append(result.timings.reward_seconds)
        stage_values["train_seconds"].append(result.timings.train_seconds)
        stage_values["iteration_seconds"].append(result.timings.iteration_seconds)
        stage_values["response_tokens"].append(float(result.response_token_count))
        rollout_timings = controller.rollout_worker.last_rollout_timings
        if rollout_timings is None:
            raise RuntimeError("tiny HF benchmark expected rollout stage timings")
        generation = rollout_timings.generation
        stage_values["prefill_seconds"].append(generation.prefill_seconds)
        stage_values["decode_seconds"].append(generation.decode_seconds)
        stage_values["prefill_forward_calls"].append(float(generation.prefill_forward_calls))
        stage_values["decode_forward_calls"].append(float(generation.decode_forward_calls))
        stage_values["old_logprob_forward_seconds"].append(rollout_timings.old_logprob_forward_seconds)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
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
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    prompt_lengths = tuple(int(part) for part in args.prompt_lengths.split(",") if part) if args.prompt_lengths else None
    config = BenchmarkConfig(
        device=args.device,
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
        utilization_monitor = (
            GpuUtilizationMonitor(config.gpu_index) if config.sample_gpu_utilization else None
        )
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
        "benchmark": "tiny_hf_grpo_controller",
        "device": config.device,
        "model": {"architecture": "GPT2LMHeadModel", "vocab_size": 64, "n_embd": 64, "n_layer": 2, "n_head": 2},
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
    }
    for key in (
        "rollout_seconds", "reward_seconds", "train_seconds", "iteration_seconds", "response_tokens",
        "prefill_seconds", "decode_seconds", "prefill_forward_calls", "decode_forward_calls",
        "old_logprob_forward_seconds",
        "prompt_batch_count", "padded_prompt_tokens", "max_batch_padded_prompt_tokens",
        "max_batch_padded_sequence_tokens", "prompt_padding_ratio",
        "train_microbatch_count", "train_real_sequence_tokens",
        "train_padded_sequence_tokens", "train_padding_ratio",
    ):
        report[f"{key}_median"] = round(statistics.median(run[key] for run in measurements), 6)
    iteration_seconds = float(report["iteration_seconds_median"])
    response_tokens = float(report["response_tokens_median"])
    report["iterations_per_second"] = round(1 / iteration_seconds, 3) if iteration_seconds else float("inf")
    report["response_tokens_per_second"] = round(response_tokens / iteration_seconds, 3) if iteration_seconds else float("inf")
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
