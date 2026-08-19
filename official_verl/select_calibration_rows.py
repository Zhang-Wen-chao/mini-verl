"""Select a small ordered parquet slice for an auditable RL calibration.

This intentionally copies already-converted official-verl rows without
changing their prompt, reward, or provenance fields.  It is used only to make
a short calibration repeatable after a no-update rollout diagnostic has
identified fixed row positions with a non-degenerate GRPO reward group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_positions(value: str) -> list[int]:
    positions = [int(part) for part in value.split(",")]
    if not positions or any(position < 0 for position in positions):
        raise ValueError("--positions must be one or more non-negative comma-separated integers")
    if len(set(positions)) != len(positions):
        raise ValueError("--positions must not contain duplicates")
    return positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--positions", required=True, help="Ordered zero-based parquet row positions, e.g. 0,1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.audit_output.exists():
        raise SystemExit("refusing to overwrite an existing output or audit file")
    if not args.input.is_file():
        raise SystemExit(f"missing input parquet: {args.input}")
    try:
        import pyarrow.parquet as pq
        from datasets import Dataset
    except ImportError as error:  # pragma: no cover - exercised in locked runtime
        raise SystemExit("pyarrow and datasets are required") from error

    positions = parse_positions(args.positions)
    rows = pq.read_table(args.input).to_pylist()
    if max(positions) >= len(rows):
        raise SystemExit(f"input has {len(rows)} rows but requested position {max(positions)}")
    selected = [rows[position] for position in positions]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(selected).to_parquet(str(args.output))
    args.audit_output.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "selection": "explicit ordered zero-based parquet positions",
                "positions": positions,
                "selected_rows": len(selected),
                "purpose": "two-step no-shuffle GRPO reward-signal calibration",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(selected)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
