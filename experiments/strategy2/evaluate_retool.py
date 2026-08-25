#!/usr/bin/env python3
"""Evaluate a ReTool policy on held-out math problems.

This mirrors the multi-turn protocol used by Strategy 2 training, but never
updates model weights.  It is deliberately independent from Slime/Ray so each
base or converted checkpoint is evaluated under identical prompts, decoding,
tool sandbox, and context limits.

Run inside ``slime-dev`` with ``PYTHONPATH`` containing this directory and
``/root/slime``.  The sibling ``generate_with_retool`` and ``tool_sandbox``
modules are the exact versions used for the Strategy 2 training runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_with_retool import (
    format_conversation_with_tools,
    postprocess_predictions,
    postprocess_responses,
)
from answer_protocol import extract_final_answer, scoreable_answer_text
from quality_reward import normalize_markdown_code
from tool_sandbox import TOOL_CONFIGS, tool_registry

try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score
except ImportError as exc:  # pragma: no cover - environment-owned dependency
    raise RuntimeError("Run this script inside the Slime container.") from exc


ERROR_MARKERS = re.compile(
    r"(?:^|\n)(?:Error|Errors|Traceback|Exception|SyntaxError|ImportError|NameError):|"
    r"Import of .+ is not allowed",
    re.IGNORECASE,
)


@dataclass
class EvaluationRecord:
    index: int
    label: str
    prediction: str
    correct: bool
    score: float
    terminal_status: str
    tool_calls: int
    markdown_fenced_tool_calls: int
    tool_successes: int
    tool_failures: int
    invalid_actions: int
    response_tokens: int
    wall_seconds: float
    response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model directory")
    parser.add_argument("--data", required=True, help="JSONL held-out data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-problems", type=int, default=0, help="0 evaluates every row")
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--tool-timeout-seconds",
        type=int,
        default=20,
        help="Per-tool timeout for this evaluation process only (default: 20).",
    )
    parser.add_argument(
        "--normalize-markdown-code",
        action="store_true",
        help=(
            "Remove only an outer ```py/```python Markdown fence in a tool JSON code field. "
            "Disabled by default to reproduce the strict training protocol."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def prompt_from_row(row: dict[str, Any]) -> str:
    messages = row["prompt"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a non-empty prompt message list.")
    user_messages = [message["content"] for message in messages if message.get("role") == "user"]
    if len(user_messages) != 1:
        raise ValueError(f"Expected exactly one user prompt, got {len(user_messages)}.")
    return user_messages[0]


def is_tool_success(observation: str) -> bool:
    return not bool(ERROR_MARKERS.search(observation))


def generate_turn(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    text: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)
    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        generation_kwargs["do_sample"] = False
    with torch.inference_mode():
        generated = model.generate(**generation_kwargs)
    generated_ids = generated[0, input_ids.shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=False), int(generated_ids.numel())


async def evaluate_one(
    *,
    index: int,
    row: dict[str, Any],
    model: AutoModelForCausalLM,
    tokenizer: Any,
    args: argparse.Namespace,
) -> EvaluationRecord:
    task_prompt = prompt_from_row(row)
    label = str(row["label"])
    prompt = format_conversation_with_tools(task_prompt, tools=tool_registry.get_tool_specs())
    response = ""
    tool_calls = markdown_fenced_tool_calls = tool_successes = tool_failures = invalid_actions = response_tokens = 0
    terminal_status = "turn_limit"
    started = perf_counter()

    for _turn in range(min(args.max_turns, TOOL_CONFIGS["max_turns"])):
        context_tokens = tokenizer(prompt + response, add_special_tokens=False).input_ids
        remaining = args.max_context_tokens - len(context_tokens)
        if remaining <= 0:
            terminal_status = "context_limit"
            break
        turn_text, turn_tokens = generate_turn(
            model,
            tokenizer,
            prompt + response,
            max_new_tokens=min(args.max_new_tokens, remaining),
            temperature=args.temperature,
        )
        response += postprocess_responses(turn_text)
        response_tokens += turn_tokens
        action, content = postprocess_predictions(turn_text)
        if action == "answer":
            terminal_status = "answer"
            break
        if action != "code":
            invalid_actions += 1
            response += (
                "\nMy previous action is invalid. If I want to execute code, I should put the code "
                "between <code> and </code>. If I want to give the final answer, I should use "
                "one boxed final value and no tool call. Let me try again.\n"
            )
            continue

        tool_calls += 1
        raw_code = content.strip()
        normalized_code, was_markdown_fenced = normalize_markdown_code(raw_code)
        markdown_fenced_tool_calls += int(was_markdown_fenced)
        code = normalized_code if args.normalize_markdown_code else raw_code
        observation = await tool_registry.execute_tool("code_interpreter", {"code": code})
        if is_tool_success(observation):
            tool_successes += 1
        else:
            tool_failures += 1
        response += f"\n\n<interpreter>\n{observation}\n</interpreter>\n\n"
    else:
        terminal_status = "turn_limit"

    final_answer = extract_final_answer(response)
    result = compute_score(scoreable_answer_text(final_answer), label, strict_box_verify=True)
    prediction = str(result.get("pred") or "")
    score = float(result.get("score", 0.0))
    return EvaluationRecord(
        index=index,
        label=label,
        prediction=prediction,
        correct=bool(result.get("acc", False)),
        score=score,
        terminal_status=terminal_status,
        tool_calls=tool_calls,
        markdown_fenced_tool_calls=markdown_fenced_tool_calls,
        tool_successes=tool_successes,
        tool_failures=tool_failures,
        invalid_actions=invalid_actions,
        response_tokens=response_tokens,
        wall_seconds=perf_counter() - started,
        response=response,
    )


def summarize(records: list[EvaluationRecord], config: dict[str, Any]) -> dict[str, Any]:
    total = len(records)
    if not total:
        raise ValueError("No evaluation rows were processed.")
    tool_calls = sum(record.tool_calls for record in records)
    tool_successes = sum(record.tool_successes for record in records)
    return {
        "config": config,
        "problem_count": total,
        "accuracy": sum(record.correct for record in records) / total,
        "correct_count": sum(record.correct for record in records),
        "mean_score": sum(record.score for record in records) / total,
        "tool_use_rate": sum(record.tool_calls > 0 for record in records) / total,
        "mean_tool_calls": tool_calls / total,
        "markdown_fenced_tool_call_rate": sum(record.markdown_fenced_tool_calls for record in records) / tool_calls
        if tool_calls
        else 0.0,
        "tool_success_rate": tool_successes / tool_calls if tool_calls else 0.0,
        "tool_failure_rate": sum(record.tool_failures for record in records) / tool_calls if tool_calls else 0.0,
        "mean_invalid_actions": sum(record.invalid_actions for record in records) / total,
        "mean_response_tokens": sum(record.response_tokens for record in records) / total,
        "mean_wall_seconds": sum(record.wall_seconds for record in records) / total,
        "terminal_status_counts": {
            status: sum(record.terminal_status == status for record in records)
            for status in sorted({record.terminal_status for record in records})
        },
    }


async def main_async(args: argparse.Namespace) -> None:
    if args.tool_timeout_seconds <= 0:
        raise ValueError("--tool-timeout-seconds must be positive.")
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    if args.max_problems:
        rows = rows[: args.max_problems]

    # Training permits long-running symbolic jobs (120 seconds).  Evaluation
    # needs a bounded per-example budget so a single malformed agent program
    # cannot dominate every arm's wall time.  This mutates only the evaluator's
    # in-process registry; it neither edits the shared sandbox source nor
    # affects a training job in another process.
    tool_registry.python_sandbox.timeout = args.tool_timeout_seconds

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    model.eval()

    records: list[EvaluationRecord] = []
    records_path = output_dir / "records.jsonl"
    with records_path.open("w") as handle:
        for index, row in enumerate(rows):
            record = await evaluate_one(index=index, row=row, model=model, tokenizer=tokenizer, args=args)
            records.append(record)
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index + 1}/{len(rows)}] correct={record.correct} tools={record.tool_calls} "
                f"tool_success={record.tool_successes} status={record.terminal_status}",
                flush=True,
            )

    config = vars(args).copy()
    (output_dir / "summary.json").write_text(json.dumps(summarize(records, config), indent=2) + "\n")


if __name__ == "__main__":
    main_args = parse_args()
    asyncio.run(main_async(main_args))
