#!/usr/bin/env python3
"""Summarize per-step one-step-off-policy timing fields from a VeRL log."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"(?:^|\s)step:(\d+)\s+-")
METRICS = (
    "timing_s/generate_async",
    "timing_s/sync_rollout_weights",
    "timing_s/ref",
    "timing_s/update_actor",
    "timing_s/step",
    "response_length/mean",
    "response_length/clip_ratio",
    "perf/throughput",
    "perf/mfu/actor",
)


def _value(line: str, key: str) -> float | None:
    match = re.search(re.escape(key) + r":(?:np\.float64\()?([-+0-9.eE]+)", line)
    return float(match.group(1)) if match else None


def parse(log_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI.sub("", raw_line)
        step = STEP.search(line)
        if not step:
            continue
        row: Dict[str, float] = {"step": float(step.group(1))}
        for metric in METRICS:
            value = _value(line, metric)
            if value is not None:
                row[metric] = value
        if "timing_s/step" in row:
            rows.append(row)
    return rows


def _summary(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)

    def percentile(q: float) -> float:
        return ordered[round((len(ordered) - 1) * q)]

    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 4),
        "p50": round(percentile(0.50), 4),
        "p90": round(percentile(0.90), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def analyze(log_path: Path) -> Dict[str, object]:
    rows = parse(log_path)
    if not rows:
        raise ValueError(f"No step metrics found in {log_path}")
    metrics = {}
    for metric in METRICS:
        values = [row[metric] for row in rows if metric in row]
        if values:
            metrics[metric] = _summary(values)
    stage_sum = [
        sum(row[key] for key in ("timing_s/generate_async", "timing_s/sync_rollout_weights", "timing_s/ref", "timing_s/update_actor"))
        for row in rows
    ]
    step_values = [row["timing_s/step"] for row in rows]
    return {
        "log_path": str(log_path),
        "steps": len(rows),
        "first_step": int(rows[0]["step"]),
        "last_step": int(rows[-1]["step"]),
        "inclusive_timing_summary_seconds": metrics,
        "sum_of_four_major_inclusive_stages_seconds": _summary(stage_sum),
        "mean_inclusive_stage_sum_minus_step_seconds": round(statistics.fmean(stage_sum) - statistics.fmean(step_values), 4),
        "interpretation": (
            "generate_async, sync_rollout_weights, ref and update_actor are inclusive timings in a controlled one-step-overlap pipeline. "
            "Their sum exceeds wall-clock step time when work overlaps; do not treat it as a serial latency breakdown."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_log", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.train_log)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
