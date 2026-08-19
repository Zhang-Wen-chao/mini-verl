"""Prepare an auditable OpenR1-Math subset in the official verl rule-reward schema.

The raw OpenR1 default split is treated as immutable input.  This program only
writes a new parquet subset and a JSON audit report.  It deliberately uses the
upstream ``math_reward`` contract: prompt the model to finish in ``\boxed{}``
and keep the raw OpenR1 ``answer`` as the rule-reward ground truth.  The prompt
is concise because the Qwen3.5 chat template's default thinking mode otherwise
consumes the rollout limit before reaching the final answer.

Before selecting rows it rejects explicit MATH/GSM8K sources and exact matches
of normalized problem text against the supplied frozen evaluation sets.  The
selection is deterministic, stratified by ``problem_type``, and memory bounded
by the requested subset size rather than the full source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

# ``data_source`` is the pinned verl reward-router key, not a provenance label.
# The chosen revision dispatches this key to ``math_reward.compute_score``.
REWARD_DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
RAW_DATA_SOURCE = "open-r1/OpenR1-Math-220k"
INSTRUCTION = (
    "Solve in at most three short sentences. Do not restate the problem, describe a plan, "
    "or add any extra commentary. End the response with the final answer exactly as \\boxed{...}."
)
SOURCE_BLOCKLIST = ("math", "gsm8k")


def normalize_problem(text: str) -> str:
    """Return a conservative exact-match key for a math problem.

    This is intentionally not fuzzy deduplication: only Unicode-normalized
    whitespace/case differences are considered equal, so the audit is easy to
    reproduce and explain.
    """

    return " ".join(text.casefold().split())


def source_is_blocked(source: str) -> bool:
    lowered = source.casefold()
    return any(token in lowered for token in SOURCE_BLOCKLIST)


def stable_rank(seed: int, uuid: str, problem: str) -> int:
    payload = f"{seed}\0{uuid}\0{normalize_problem(problem)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def proportional_quotas(counts: Mapping[str, int], limit: int) -> dict[str, int]:
    """Allocate exactly ``limit`` slots proportionally with deterministic ties."""

    total = sum(counts.values())
    if limit <= 0:
        raise ValueError("limit must be positive")
    if total < limit:
        raise ValueError(f"only {total} eligible rows remain for requested limit {limit}")

    base = {kind: min(count, (count * limit) // total) for kind, count in counts.items()}
    remaining = limit - sum(base.values())
    order = sorted(
        counts,
        key=lambda kind: (
            -((counts[kind] * limit) % total),
            kind,
        ),
    )
    for kind in order:
        if remaining == 0:
            break
        if base[kind] < counts[kind]:
            base[kind] += 1
            remaining -= 1
    if remaining:
        raise ValueError("could not allocate all requested stratified slots")
    return base


def is_eligible(row: Mapping[str, Any], evaluation_keys: set[str]) -> tuple[bool, str | None]:
    problem = row.get("problem")
    answer = row.get("answer")
    source = row.get("source")
    if not isinstance(problem, str) or not problem.strip():
        return False, "missing_problem"
    if not isinstance(answer, str) or not answer.strip():
        return False, "missing_answer"
    if not isinstance(source, str) or not source.strip():
        return False, "missing_source"
    if source_is_blocked(source):
        return False, "blocked_source"
    if normalize_problem(problem) in evaluation_keys:
        return False, "evaluation_exact_duplicate"
    return True, None


def official_verl_row(row: Mapping[str, Any], *, split: str, original_index: int) -> dict[str, Any]:
    problem = str(row["problem"]).strip()
    answer = str(row["answer"]).strip()
    return {
        "data_source": REWARD_DATA_SOURCE,
        "prompt": [{"role": "user", "content": f"{problem}\n\n{INSTRUCTION}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "original_index": original_index,
            "uuid": row.get("uuid"),
            "raw_data_source": RAW_DATA_SOURCE,
            "source": row.get("source"),
            "problem_type": row.get("problem_type"),
            "question_type": row.get("question_type"),
        },
    }


def iter_parquet_rows(paths: Sequence[Path], columns: Sequence[str]) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - exercised on L20 runtime
        raise SystemExit("pyarrow is required to read the downloaded parquet files") from error

    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=list(columns), batch_size=1024):
            yield from batch.to_pylist()


def evaluation_keys(paths_and_columns: Sequence[tuple[Path, str]]) -> set[str]:
    keys: set[str] = set()
    for path, column in paths_and_columns:
        for row in iter_parquet_rows([path], [column]):
            value = row.get(column)
            if isinstance(value, str) and value.strip():
                keys.add(normalize_problem(value))
    return keys


def scan_eligible(
    paths: Sequence[Path], evaluation_problem_keys: set[str]
) -> tuple[Counter[str], Counter[str]]:
    counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    for row in iter_parquet_rows(paths, ["problem", "answer", "source", "problem_type"]):
        accepted, reason = is_eligible(row, evaluation_problem_keys)
        if not accepted:
            assert reason is not None
            rejected[reason] += 1
            continue
        kind = row.get("problem_type")
        counts[str(kind) if isinstance(kind, str) and kind else "unknown"] += 1
    return counts, rejected


def select_rows(
    paths: Sequence[Path],
    *,
    evaluation_problem_keys: set[str],
    quotas: Mapping[str, int],
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Keep the lowest stable ranks per problem type using bounded heaps."""

    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {kind: [] for kind in quotas}
    columns = ["problem", "answer", "source", "problem_type", "question_type", "uuid"]
    original_index = 0
    for row in iter_parquet_rows(paths, columns):
        accepted, _ = is_eligible(row, evaluation_problem_keys)
        if accepted:
            kind_value = row.get("problem_type")
            kind = str(kind_value) if isinstance(kind_value, str) and kind_value else "unknown"
            quota = quotas.get(kind, 0)
            if quota:
                uuid = row.get("uuid")
                rank = stable_rank(seed, str(uuid) if uuid is not None else str(original_index), str(row["problem"]))
                item = (-rank, original_index, row)
                heap = heaps[kind]
                if len(heap) < quota:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
        original_index += 1

    selected: list[tuple[int, dict[str, Any]]] = []
    for kind, quota in quotas.items():
        heap = heaps[kind]
        if len(heap) != quota:
            raise ValueError(f"selection for {kind!r} has {len(heap)} rows, expected {quota}")
        selected.extend((index, row) for _negative_rank, index, row in heap)
    return sorted(selected, key=lambda item: (str(item[1].get("problem_type")), item[0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--math-test", type=Path, required=True)
    parser.add_argument("--gsm8k-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.output.exists() or args.audit_output.exists():
        raise SystemExit("refusing to overwrite an existing output or audit file")

    source_paths = sorted(args.input_dir.glob("*.parquet"))
    if not source_paths:
        raise SystemExit(f"no parquet files under {args.input_dir}")
    for path in (args.math_test, args.gsm8k_test):
        if not path.is_file():
            raise SystemExit(f"missing evaluation parquet: {path}")

    eval_keys = evaluation_keys([(args.math_test, "problem"), (args.gsm8k_test, "question")])
    eligible_counts, rejected_counts = scan_eligible(source_paths, eval_keys)
    quotas = proportional_quotas(eligible_counts, args.limit)
    selected = select_rows(source_paths, evaluation_problem_keys=eval_keys, quotas=quotas, seed=args.seed)

    try:
        from datasets import Dataset
    except ImportError as error:  # pragma: no cover - exercised on L20 runtime
        raise SystemExit("datasets is required to write official verl parquet files") from error

    rows = [official_verl_row(row, split=args.split, original_index=index) for index, row in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(args.output))
    audit = {
        "raw_data_source": RAW_DATA_SOURCE,
        "reward_data_source": REWARD_DATA_SOURCE,
        "seed": args.seed,
        "limit": args.limit,
        "source_files": [str(path) for path in source_paths],
        "evaluation_problem_key_count": len(eval_keys),
        "eligible_by_problem_type": dict(sorted(eligible_counts.items())),
        "rejected_by_reason": dict(sorted(rejected_counts.items())),
        "selected_by_problem_type": dict(sorted(Counter(str(row.get("problem_type")) for _, row in selected).items())),
        "selected_rows": len(rows),
        "reward_contract": "last \\boxed{...} compared using upstream verl.utils.reward_score.math_reward",
        "prompt_contract": INSTRUCTION,
    }
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} official-verl rows to {args.output}")
    print(f"wrote audit report to {args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
