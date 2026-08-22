"""Convert frozen MATH-lighteval test rows into official verl parquet."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _openr1_contract() -> tuple[str, str]:
    """Load the sibling contract when invoked as a script or file module."""

    path = Path(__file__).with_name("prepare_openr1_math.py")
    spec = importlib.util.spec_from_file_location("official_verl_openr1_contract", path)
    if spec is None or spec.loader is None:  # pragma: no cover - impossible for a regular file
        raise RuntimeError(f"could not load shared OpenR1 contract from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.INSTRUCTION, module.REWARD_DATA_SOURCE


INSTRUCTION, REWARD_DATA_SOURCE = _openr1_contract()


def last_boxed_answer(solution: str) -> str:
    """Return the payload of the final balanced \boxed{...} answer."""

    start = solution.rfind("\\boxed{")
    if start < 0:
        raise ValueError("solution has no boxed final answer")
    index = start + len("\\boxed{")
    payload_start = index
    depth = 1
    while index < len(solution):
        char = solution[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return solution[payload_start:index].strip()
        index += 1
    raise ValueError("unbalanced boxed final answer")


def official_row(row: dict[str, Any], original_index: int) -> dict[str, Any]:
    problem = str(row["problem"]).strip()
    return {
        "data_source": REWARD_DATA_SOURCE,
        "prompt": [{"role": "user", "content": f"{problem}\n\n{INSTRUCTION}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": last_boxed_answer(str(row["solution"]))},
        "extra_info": {
            "split": "test",
            "original_index": original_index,
            "raw_data_source": REWARD_DATA_SOURCE,
            "level": row.get("level"),
            "problem_type": row.get("type"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.output.exists() or args.audit_output.exists():
        raise SystemExit("refusing to overwrite an existing output or audit file")
    if not args.input.is_file():
        raise SystemExit(f"missing input parquet: {args.input}")

    try:
        import pyarrow.parquet as pq
        from datasets import Dataset
    except ImportError as error:  # pragma: no cover - exercised on L20 runtime
        raise SystemExit("pyarrow and datasets are required") from error

    rows = pq.read_table(args.input, columns=["problem", "solution", "level", "type"]).to_pylist()
    if len(rows) < args.limit:
        raise SystemExit(f"only {len(rows)} rows available for requested limit {args.limit}")
    converted = [official_row(row, index) for index, row in enumerate(rows[: args.limit])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(converted).to_parquet(str(args.output))
    args.audit_output.write_text(
        json.dumps(
            {
                "raw_data_source": REWARD_DATA_SOURCE,
                "reward_data_source": REWARD_DATA_SOURCE,
                "input": str(args.input),
                "selection": "first_rows_in_frozen_parquet_order",
                "selected_rows": len(converted),
                "reward_contract": "last boxed answer compared using upstream math_reward",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(converted)} official-verl validation rows to {args.output}")
    print(f"wrote audit report to {args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
