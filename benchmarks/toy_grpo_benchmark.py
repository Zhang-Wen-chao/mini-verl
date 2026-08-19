"""Benchmark the full synchronous Controller iteration on the toy GRPO workload.

This includes rollout, rule reward and policy update. It is a framework
correctness benchmark, not a serving or large-model throughput benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from examples.toy_grpo_train import run
from mini_verl.observability import CudaMemoryMonitor, GpuUtilizationMonitor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--sample-gpu-utilization",
        action="store_true",
        help="sample device-level nvidia-smi metrics; use only on an idle/exclusive physical GPU",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="physical nvidia-smi GPU index for --sample-gpu-utilization",
    )
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("iterations/repeats must be positive and warmup must be non-negative")

    for repeat in range(args.warmup):
        run(device=args.device, iterations=args.iterations, seed=10_000 + repeat)

    durations = []
    final_iteration_durations = []
    peak_allocated_bytes = []
    peak_reserved_bytes = []
    utilization_means = []
    utilization_maxima = []
    memory_used_maxima = []
    utilization_sample_counts = []
    for repeat in range(args.repeats):
        monitor = CudaMemoryMonitor(args.device or "cpu")
        utilization_monitor = (
            GpuUtilizationMonitor(args.gpu_index) if args.sample_gpu_utilization else None
        )
        monitor.start()
        if utilization_monitor is not None:
            utilization_monitor.start()
        started = time.perf_counter()
        result = run(device=args.device, iterations=args.iterations, seed=20_000 + repeat)
        durations.append(time.perf_counter() - started)
        final_iteration_durations.append(result.final_iteration_seconds)
        memory = monitor.stop()
        utilization = utilization_monitor.stop() if utilization_monitor is not None else None
        if memory is not None:
            peak_allocated_bytes.append(memory.max_allocated_bytes)
            peak_reserved_bytes.append(memory.max_reserved_bytes)
        if utilization is not None:
            utilization_means.append(utilization.mean_utilization_percent)
            utilization_maxima.append(utilization.max_utilization_percent)
            memory_used_maxima.append(utilization.max_memory_used_mib)
            utilization_sample_counts.append(utilization.sample_count)

    median_duration = statistics.median(durations)
    report = {
        "benchmark": "toy_grpo_controller",
        "device": args.device or "auto",
        "iterations": args.iterations,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "run_seconds_median": round(median_duration, 6),
        "iteration_seconds_median": round(statistics.median(final_iteration_durations), 6),
        "iterations_per_second": round(args.iterations / median_duration, 3),
    }
    if peak_allocated_bytes:
        report.update({
            "cuda_max_allocated_bytes_median": int(statistics.median(peak_allocated_bytes)),
            "cuda_max_reserved_bytes_median": int(statistics.median(peak_reserved_bytes)),
        })
    if utilization_means:
        report.update({
            "gpu_utilization_scope": "device_level_nvidia_smi",
            "gpu_physical_index": args.gpu_index,
            "gpu_utilization_interval_seconds": 0.1,
            "gpu_utilization_mean_percent_median": round(statistics.median(utilization_means), 3),
            "gpu_utilization_max_percent_max": max(utilization_maxima),
            "gpu_memory_used_max_mib_max": max(memory_used_maxima),
            "gpu_utilization_sample_count_median": int(statistics.median(utilization_sample_counts)),
        })
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
