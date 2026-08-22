"""Stable data and reward contract for the first official-verl experiment.

This module is deliberately independent of upstream verl internals.  It converts
GSM8K's public JSONL records into a minimal prompt/answer JSONL schema and scores
only the final boxed integer emitted by a rollout.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

FINAL_ANSWER_MARKER = "####"
BOXED_INTEGER = re.compile(r"\\boxed\{\s*([+-]?\d[\d,]*)\s*\}")

PROMPT_TEMPLATE = """Solve the following math problem. Show your reasoning.
End the response with exactly one final answer in the form \\boxed{{integer}}.

Problem:
{question}
"""


def normalize_integer(value: str) -> str | None:
    """Normalize an integer spelling, returning None for malformed values."""
    compact = value.strip().replace(",", "")
    if not re.fullmatch(r"[+-]?\d+", compact):
        return None
    return str(int(compact))


def gsm8k_final_answer(solution: str) -> str:
    """Extract the canonical GSM8K answer after its final `####` marker."""
    if FINAL_ANSWER_MARKER not in solution:
        raise ValueError("GSM8K answer is missing the final '####' marker")
    answer = normalize_integer(solution.rsplit(FINAL_ANSWER_MARKER, 1)[1])
    if answer is None:
        raise ValueError("GSM8K final answer must be an integer")
    return answer


def boxed_final_answer(response: str) -> str | None:
    """Extract the last boxed integer, rejecting extra text after it."""
    matches = list(BOXED_INTEGER.finditer(response))
    if not matches:
        return None
    final = matches[-1]
    if response[final.end() :].strip():
        return None
    return normalize_integer(final.group(1))


def exact_boxed_integer_reward(response: str, expected_answer: str) -> float:
    """Return a deterministic 0/1 reward without rewarding a malformed answer."""
    expected = normalize_integer(expected_answer)
    if expected is None:
        raise ValueError("expected_answer must be an integer")
    return float(boxed_final_answer(response) == expected)


def convert_records(records: Iterable[dict[str, object]]) -> Iterator[dict[str, str]]:
    """Convert public GSM8K `{question, answer}` records to experiment rows."""
    for index, record in enumerate(records):
        question = record.get("question")
        solution = record.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"record {index}: missing non-empty question")
        if not isinstance(solution, str):
            raise ValueError(f"record {index}: missing string answer")
        yield {
            "prompt": PROMPT_TEMPLATE.format(question=question.strip()),
            "answer": gsm8k_final_answer(solution),
        }


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def write_jsonl(rows: Iterable[dict[str, str]], path: Path, limit: int | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if limit is not None and count >= limit:
                break
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
