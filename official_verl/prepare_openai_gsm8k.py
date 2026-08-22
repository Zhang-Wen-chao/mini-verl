"""Prepare OpenAI's public GSM8K JSONL using verl's upstream row contract.

The official `examples/data_preprocess/gsm8k.py` downloads the Hugging Face
mirror and writes this same `data_source/prompt/ability/reward_model/extra_info`
shape. This adapter accepts the canonical OpenAI `train.jsonl` and `test.jsonl`
when Hugging Face is unavailable on an isolated training host.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

DATA_SOURCE = "openai/gsm8k"
INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
FINAL_ANSWER = re.compile(r"####\s*(-?[0-9.,]+)\s*$")


def extract_solution(solution: str) -> str:
    """Mirror verl's GSM8K extractor while rejecting malformed trailing text."""
    match = FINAL_ANSWER.search(solution)
    if match is None:
        raise ValueError("GSM8K solution must end with '#### <numeric answer>'")
    answer = match.group(1).replace(",", "")
    if not re.fullmatch(r"-?[0-9.]+", answer):
        raise ValueError("GSM8K final answer is not numeric")
    return answer


def convert_record(record: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    question = record.get("question")
    answer = record.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"record {index}: missing non-empty question")
    if not isinstance(answer, str):
        raise ValueError(f"record {index}: missing string answer")
    question_raw = question.strip()
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": f"{question_raw} {INSTRUCTION}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": extract_solution(answer)},
        "extra_info": {
            "split": split,
            "index": index,
            "answer": answer,
            "question": question_raw,
        },
    }


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield record


def convert_records(records: Iterable[dict[str, Any]], *, split: str, limit: int | None) -> Iterator[dict[str, Any]]:
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            return
        yield convert_record(record, split=split, index=index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    try:
        from datasets import Dataset
    except ImportError as error:
        raise SystemExit("Install `datasets` to write official verl parquet files.") from error

    rows = list(convert_records(read_jsonl(args.input), split=args.split, limit=args.limit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(args.output))
    print(f"wrote {len(rows)} official-verl GSM8K rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
