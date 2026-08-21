#!/usr/bin/env python3
"""Audit legacy versus conservative normalized rewards in dumped rollouts."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from normalized_math_reward import compute_score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()

    files = sorted(glob.glob(str(args.rollout_dir / "*.jsonl")), key=lambda p: int(Path(p).stem))
    changes, by_step, groups = [], Counter(), defaultdict(lambda: {"legacy": [], "normalized": []})
    totals = Counter()
    for filename in files:
        step = int(Path(filename).stem)
        with open(filename, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                # The saved score is the exact reward used by the historical worker.
                # This keeps the audit independent of the heavyweight VeRL runtime.
                legacy = float(record["score"])
                normalized = compute_score(record["output"], record["gts"], lambda _output, _target: legacy)
                totals["samples"] += 1
                totals[f"legacy_{int(legacy)}"] += 1
                totals[f"normalized_{int(normalized)}"] += 1
                key = (step, record["input"])
                groups[key]["legacy"].append(legacy)
                groups[key]["normalized"].append(normalized)
                if legacy != normalized:
                    totals["changed"] += 1
                    by_step[step] += 1
                    if len(changes) < args.examples:
                        changes.append({
                            "step": step, "line": line_number, "legacy": legacy, "normalized": normalized,
                            "ground_truth": record["gts"], "output": record["output"],
                        })
    changed_groups = sum(value["legacy"] != value["normalized"] for value in groups.values())
    group_effect = Counter()
    for value in groups.values():
        legacy_mixed = len(set(value["legacy"])) > 1
        normalized_mixed = len(set(value["normalized"])) > 1
        group_effect["total"] += 1
        group_effect[f"legacy_{'mixed' if legacy_mixed else 'degenerate'}"] += 1
        group_effect[f"normalized_{'mixed' if normalized_mixed else 'degenerate'}"] += 1
        if legacy_mixed != normalized_mixed:
            group_effect["mixed_status_changed"] += 1
    report = {
        "rollout_dir": str(args.rollout_dir),
        "files": len(files),
        "samples": totals["samples"],
        "legacy_positive": totals["legacy_1"],
        "normalized_positive": totals["normalized_1"],
        "changed_samples": totals["changed"],
        "changed_groups": changed_groups,
        "group_effect": dict(group_effect),
        "changed_by_step": dict(sorted(by_step.items())),
        "examples": changes,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "examples"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
