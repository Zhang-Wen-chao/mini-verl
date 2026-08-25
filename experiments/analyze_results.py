"""Analyze overnight experiment JSONs and print a comparison summary.

Reads every agent_*.json and stab_*.json under results/ and prints:
  - final mean_reward, mean tool_rate, clip_fraction per run
  - reward trajectory (first/mid/last)
No plotting dependency: plain text tables, plus optional CSV export.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_results(results_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            out[path.stem] = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"!! could not load {path.name}: {e}")
    return out


def summarize(name: str, data: dict) -> dict:
    records = data.get("records", [])
    if not records:
        return {"name": name, "iters": 0}
    first, mid, last = records[0], records[len(records) // 2], records[-1]
    cfg = data.get("config", {})
    return {
        "name": name,
        "iters": len(records),
        "mode": cfg.get("mode", cfg.get("clip_reward") is not None and "stab" or "?"),
        "reward_first": first.get("mean_reward"),
        "reward_mid": mid.get("mean_reward"),
        "reward_last": last.get("mean_reward"),
        "tool_rate_last": last.get("tool_rate"),
        "tool_success_last": last.get("tool_success_rate"),
        "clip_last": last.get("clip_fraction"),
        "mean_len_last": last.get("mean_response_tokens"),
        "mean_reward_avg": sum(r.get("mean_reward", 0) or 0 for r in records) / len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results", help="results dir")
    parser.add_argument("--csv", default=None, help="optional CSV output path")
    args = parser.parse_args()

    results_dir = Path(args.results)
    all_results = load_results(results_dir)
    rows = [summarize(name, data) for name, data in sorted(all_results.items())]

    print(f"\n{'run':<22}{'iters':>6}{'r_first':>9}{'r_mid':>9}{'r_last':>9}{'avg':>7}{'tool':>7}{'clip':>8}{'len':>7}")
    print("-" * 95)
    for r in rows:
        if r["iters"] == 0:
            print(f"{r['name']:<22}  (no records)")
            continue
        print(
            f"{r['name']:<22}{r['iters']:>6}"
            f"{r['reward_first']:>9.3f}{r['reward_mid']:>9.3f}{r['reward_last']:>9.3f}"
            f"{r['mean_reward_avg']:>7.3f}"
            f"{r['tool_rate_last'] if r['tool_rate_last'] is not None else float('nan'):>7.3f}"
            f"{r['clip_last'] if r['clip_last'] is not None else float('nan'):>8.3f}"
            f"{r['mean_len_last'] if r['mean_len_last'] is not None else float('nan'):>7.1f}"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
