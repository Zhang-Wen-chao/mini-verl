#!/usr/bin/env python3
"""Re-score completed Strategy 2 records with the current answer protocol.

This is deliberately offline: it determines whether an evaluator/parser change
would alter results already collected, without spending additional GPU time.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from answer_protocol import extract_final_answer, scoreable_answer_text
from slime.rollout.rm_hub.math_dapo_utils import compute_score


ARMS = ("base", "outcome_reward", "process_reward", "quality_process_reward_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Evaluation directory containing results/<arm>/records.jsonl")
    parser.add_argument("--output", help="Optional JSON report path; defaults to <run-root>/rescored_summary.json")
    return parser.parse_args()


def analyze_arm(records_path: Path) -> dict[str, object]:
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    rescored: list[dict[str, object]] = []
    for record in records:
        answer = extract_final_answer(record["response"])
        result = compute_score(scoreable_answer_text(answer), str(record["label"]), strict_box_verify=True)
        rescored.append(
            {
                "index": record["index"],
                "label": str(record["label"]),
                "terminal_status": record["terminal_status"],
                "previous_prediction": record["prediction"],
                "extracted_answer": answer or "",
                "previous_correct": bool(record["correct"]),
                "rescored_correct": bool(result.get("acc", False)),
            }
        )

    recovered = [record for record in rescored if record["rescored_correct"] and not record["previous_correct"]]
    return {
        "problem_count": len(records),
        "previous_correct_count": sum(record["previous_correct"] for record in rescored),
        "rescored_correct_count": sum(record["rescored_correct"] for record in rescored),
        "extracted_answer_count": sum(bool(record["extracted_answer"]) for record in rescored),
        "terminal_status_counts": dict(Counter(record["terminal_status"] for record in rescored)),
        "recovered_correct_records": recovered,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    report = {arm: analyze_arm(root / "results" / arm / "records.jsonl") for arm in ARMS}
    output = Path(args.output) if args.output else root / "rescored_summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for arm, metrics in report.items():
        print(
            arm,
            f"previous_correct={metrics['previous_correct_count']}",
            f"rescored_correct={metrics['rescored_correct_count']}",
            f"extracted_answers={metrics['extracted_answer_count']}",
            f"recovered={len(metrics['recovered_correct_records'])}",
        )


if __name__ == "__main__":
    main()
