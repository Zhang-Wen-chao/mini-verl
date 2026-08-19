"""Measure static prompt-length bucketing in the real HF rollout path.

The model is constructed locally and intentionally tiny.  This benchmark is a
repeatable regression workload for ``HuggingFaceRolloutWorker`` rather than a
claim about production serving throughput.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from time import perf_counter

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from mini_verl.config import RunConfig, seed_everything
from mini_verl.hf import HuggingFaceRolloutWorker, PromptExample
from mini_verl.observability import CudaMemoryMonitor, GpuUtilizationMonitor


class VariableLengthTokenizer:
    """Offline tokenizer whose prompt text encodes its exact token length."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, *, return_tensors: str, add_special_tokens: bool):
        del return_tensors, add_special_tokens
        length = int(text.removeprefix("length="))
        token_ids = [2 + index % 62 for index in range(length)]
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
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
    prompt_lengths: tuple[int, ...]
    group_size: int
    max_new_tokens: int
    rollout_batch_size: int
    bucket_prompts_by_length: bool
    rollout_max_padded_prompt_tokens: int | None
    rollout_max_padded_sequence_tokens: int | None
    sample_gpu_utilization: bool
    gpu_index: int

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.warmup < 0 or self.repeats <= 0:
            raise ValueError("iterations/repeats must be positive and warmup must be non-negative")
        if not self.prompt_lengths or min(self.prompt_lengths) <= 0:
            raise ValueError("prompt_lengths must contain positive integers")
        if self.group_size <= 0 or self.max_new_tokens <= 0 or self.rollout_batch_size <= 0:
            raise ValueError("group_size, max_new_tokens, and rollout_batch_size must be positive")
        if self.rollout_max_padded_prompt_tokens is not None and self.rollout_max_padded_prompt_tokens <= 0:
            raise ValueError("rollout_max_padded_prompt_tokens must be positive when set")
        if self.rollout_max_padded_sequence_tokens is not None and self.rollout_max_padded_sequence_tokens <= 0:
            raise ValueError("rollout_max_padded_sequence_tokens must be positive when set")


def _build_worker(config: BenchmarkConfig, *, seed: int) -> HuggingFaceRolloutWorker:
    seed_everything(RunConfig(seed=seed, device=config.device, deterministic=False))
    device = torch.device(config.device)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_positions=max(32, max(config.prompt_lengths) + config.max_new_tokens),
            n_embd=64,
            n_layer=2,
            n_head=2,
            pad_token_id=VariableLengthTokenizer.pad_token_id,
            bos_token_id=VariableLengthTokenizer.eos_token_id,
            eos_token_id=VariableLengthTokenizer.eos_token_id,
        )
    ).to(device)
    return HuggingFaceRolloutWorker(
        model=model,
        tokenizer=VariableLengthTokenizer(),
        prompts=[PromptExample(f"length={length}", {"prompt_index": index}) for index, length in enumerate(config.prompt_lengths)],
        group_size=config.group_size,
        max_new_tokens=config.max_new_tokens,
        rollout_batch_size=config.rollout_batch_size,
        bucket_prompts_by_length=config.bucket_prompts_by_length,
        rollout_max_padded_prompt_tokens=config.rollout_max_padded_prompt_tokens,
        rollout_max_padded_sequence_tokens=config.rollout_max_padded_sequence_tokens,
        do_sample=True,
        collect_generation_timings=True,
    )


def _run_once(config: BenchmarkConfig, *, seed: int) -> dict[str, float]:
    worker = _build_worker(config, seed=seed)
    device = next(worker.model.parameters()).device
    values = {
        "rollout_seconds": [],
        "response_tokens": [],
        "prefill_seconds": [],
        "decode_seconds": [],
        "prefill_forward_calls": [],
        "decode_forward_calls": [],
        "old_logprob_forward_seconds": [],
        "prompt_batch_count": [],
        "real_prompt_tokens": [],
        "padded_prompt_tokens": [],
        "max_batch_padded_prompt_tokens": [],
        "max_batch_padded_sequence_tokens": [],
        "prompt_padding_ratio": [],
    }
    for policy_version in range(config.iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = perf_counter()
        rollout = worker.rollout(policy_version=policy_version)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values["rollout_seconds"].append(perf_counter() - started)
        timings = worker.last_rollout_timings
        if timings is None:
            raise RuntimeError("benchmark expected rollout timings")
        stats = timings.prompt_batching
        values["response_tokens"].append(float(rollout.response_token_count))
        values["prefill_seconds"].append(timings.generation.prefill_seconds)
        values["decode_seconds"].append(timings.generation.decode_seconds)
        values["prefill_forward_calls"].append(float(timings.generation.prefill_forward_calls))
        values["decode_forward_calls"].append(float(timings.generation.decode_forward_calls))
        values["old_logprob_forward_seconds"].append(timings.old_logprob_forward_seconds)
        values["prompt_batch_count"].append(float(stats.batch_count))
        values["real_prompt_tokens"].append(float(stats.real_prompt_tokens))
        values["padded_prompt_tokens"].append(float(stats.padded_prompt_tokens))
        values["max_batch_padded_prompt_tokens"].append(float(stats.max_batch_padded_prompt_tokens))
        values["max_batch_padded_sequence_tokens"].append(float(stats.max_batch_padded_sequence_tokens))
        values["prompt_padding_ratio"].append(stats.padding_ratio)
    return {key: statistics.median(rows) for key, rows in values.items()}


def _parse_lengths(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split(",") if part)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--prompt-lengths must be comma-separated integers") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt-lengths", type=_parse_lengths, default=_parse_lengths("3,24,4,23,5,22,6,21"))
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--bucket-prompts-by-length", action="store_true")
    parser.add_argument("--rollout-max-padded-prompt-tokens", type=int)
    parser.add_argument("--rollout-max-padded-sequence-tokens", type=int)
    parser.add_argument("--sample-gpu-utilization", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    config = BenchmarkConfig(
        device=args.device, iterations=args.iterations, warmup=args.warmup, repeats=args.repeats,
        prompt_lengths=args.prompt_lengths, group_size=args.group_size, max_new_tokens=args.max_new_tokens,
        rollout_batch_size=args.rollout_batch_size, bucket_prompts_by_length=args.bucket_prompts_by_length,
        rollout_max_padded_prompt_tokens=args.rollout_max_padded_prompt_tokens,
        rollout_max_padded_sequence_tokens=args.rollout_max_padded_sequence_tokens,
        sample_gpu_utilization=args.sample_gpu_utilization, gpu_index=args.gpu_index,
    )
    for repeat in range(config.warmup):
        _run_once(config, seed=10_000 + repeat)

    measurements: list[dict[str, float]] = []
    allocated_peaks: list[int] = []
    reserved_peaks: list[int] = []
    utilization_means: list[float] = []
    utilization_maxima: list[int] = []
    utilization_memory_maxima: list[int] = []
    for repeat in range(config.repeats):
        memory_monitor = CudaMemoryMonitor(config.device)
        utilization_monitor = GpuUtilizationMonitor(config.gpu_index) if config.sample_gpu_utilization else None
        memory_monitor.start()
        if utilization_monitor is not None:
            utilization_monitor.start()
        measurements.append(_run_once(config, seed=20_000 + repeat))
        memory = memory_monitor.stop()
        utilization = utilization_monitor.stop() if utilization_monitor is not None else None
        if memory is not None:
            allocated_peaks.append(memory.max_allocated_bytes)
            reserved_peaks.append(memory.max_reserved_bytes)
        if utilization is not None:
            utilization_means.append(utilization.mean_utilization_percent)
            utilization_maxima.append(utilization.max_utilization_percent)
            utilization_memory_maxima.append(utilization.max_memory_used_mib)

    report: dict[str, object] = {
        "benchmark": "hf_rollout_prompt_bucketing", "device": config.device,
        "model": {"architecture": "GPT2LMHeadModel", "vocab_size": 64, "n_embd": 64, "n_layer": 2, "n_head": 2},
        "iterations_per_repeat": config.iterations, "warmup": config.warmup, "repeats": config.repeats,
        "prompt_lengths": list(config.prompt_lengths), "group_size": config.group_size,
        "max_new_tokens": config.max_new_tokens, "rollout_batch_size": config.rollout_batch_size,
        "bucket_prompts_by_length": config.bucket_prompts_by_length,
        "rollout_max_padded_prompt_tokens": config.rollout_max_padded_prompt_tokens,
        "rollout_max_padded_sequence_tokens": config.rollout_max_padded_sequence_tokens,
    }
    for key in next(iter(measurements)):
        report[f"{key}_median"] = round(statistics.median(run[key] for run in measurements), 6)
    rollout_seconds = float(report["rollout_seconds_median"])
    response_tokens = float(report["response_tokens_median"])
    report["response_tokens_per_second"] = round(response_tokens / rollout_seconds, 3) if rollout_seconds else float("inf")
    if allocated_peaks:
        report["cuda_max_allocated_bytes_median"] = int(statistics.median(allocated_peaks))
        report["cuda_max_reserved_bytes_median"] = int(statistics.median(reserved_peaks))
    if utilization_means:
        report.update({
            "gpu_utilization_scope": "device_level_nvidia_smi", "gpu_physical_index": config.gpu_index,
            "gpu_utilization_interval_seconds": 0.1,
            "gpu_utilization_mean_percent_median": round(statistics.median(utilization_means), 3),
            "gpu_utilization_max_percent_max": max(utilization_maxima),
            "gpu_memory_used_max_mib_max": max(utilization_memory_maxima),
        })
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
