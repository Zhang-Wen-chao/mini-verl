#!/usr/bin/env python3
"""Compare outcome-level GRPO advantage scales from saved rollout rewards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
import math
from pathlib import Path
from typing import Any


def _group_rewards(rollout_dir: Path) -> dict[tuple[int, str], list[float]]:
    groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    for filename in sorted(glob.glob(str(rollout_dir / "*.jsonl")), key=lambda path: int(Path(path).stem)):
        with open(filename, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                groups[(int(record["step"]), record["input"])].append(float(record["score"]))
    return groups


def advantages(rewards: list[float], normalize_by_std: bool) -> list[float]:
    """Reproduce pinned VeRL's centered reward and Bessel-corrected std."""
    mean = sum(rewards) / len(rewards)
    centered = [reward - mean for reward in rewards]
    if not normalize_by_std or len(rewards) < 2:
        return centered
    variance = sum(value * value for value in centered) / (len(rewards) - 1)
    std = math.sqrt(variance)
    return [value / (std + 1e-6) for value in centered] if std else centered


def analyze(rollout_dir: Path) -> dict[str, Any]:
    groups = _group_rewards(rollout_dir)
    group_size_histogram = Counter(map(len, groups.values()))
    positive_histogram = Counter(int(sum(rewards)) for rewards in groups.values())
    standardized: list[float] = []
    centered: list[float] = []
    scale_ratios: dict[str, list[float]] = defaultdict(list)

    for rewards in groups.values():
        with_std = advantages(rewards, normalize_by_std=True)
        without_std = advantages(rewards, normalize_by_std=False)
        standardized.extend(with_std)
        centered.extend(without_std)
        centered_l1 = sum(abs(value) for value in without_std)
        std_l1 = sum(abs(value) for value in with_std)
        if centered_l1:
            scale_ratios[str(int(sum(rewards)))].append(std_l1 / centered_l1)

    def summary(values: list[float]) -> dict[str, float]:
        return {
            "sum_abs": round(sum(abs(value) for value in values), 6),
            "mean_abs": round(sum(abs(value) for value in values) / len(values), 6),
            "rms": round(math.sqrt(sum(value * value for value in values) / len(values)), 6),
        }

    return {
        "rollout_dir": str(rollout_dir),
        "groups": len(groups),
        "samples": sum(map(len, groups.values())),
        "group_size_histogram": {str(key): value for key, value in sorted(group_size_histogram.items())},
        "positive_reward_histogram": {str(key): value for key, value in sorted(positive_histogram.items())},
        "mixed_reward_groups": sum(value for key, value in positive_histogram.items() if 0 < key < 4),
        "standard_grpo": summary(standardized),
        "dr_grpo_without_std": summary(centered),
        "mean_std_to_centered_l1_scale_by_positive_count": {
            key: round(sum(values) / len(values), 6) for key, values in sorted(scale_ratios.items())
        },
        "interpretation": (
            "Outcome-level scale before token aggregation, PPO clipping and KL; a runtime calibration must verify gradient norms."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.rollout_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
