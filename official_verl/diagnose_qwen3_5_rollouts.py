"""Run a no-update Qwen3.5 math-rollout format diagnostic with vLLM.

This is deliberately outside the trainer: it never constructs an optimizer,
FSDP worker, Ray cluster, or checkpoint.  It renders the model's native chat
template under a small set of prompt/thinking variants, generates completions,
and scores them with the same upstream ``math_reward`` used by the official
verl GRPO run.  The resulting JSONL keeps the evidence needed to decide
whether a GRPO rollout has a non-degenerate reward signal.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LEGACY_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."
CONCISE_INSTRUCTION = (
    "Solve the problem concisely. Do not describe a plan or repeat the question. "
    "End with the final answer exactly as \\boxed{...}."
)


FINAL_ONLY_INSTRUCTION = (
    "Give only the final answer. Do not show reasoning, explanation, a plan, or prose. "
    "Your entire response must be exactly one LaTeX expression of the form \\boxed{...}."
)
SHORT_SOLUTION_INSTRUCTION = (
    "Solve in at most three short sentences. Do not restate the problem, describe a plan, "
    "or add any extra commentary. End the response with the final answer exactly as \\boxed{...}."
)


def question_from_content(content: str) -> str:
    """Remove a known rollout instruction without altering the math question."""

    for instruction in (
        SHORT_SOLUTION_INSTRUCTION,
        FINAL_ONLY_INSTRUCTION,
        CONCISE_INSTRUCTION,
        LEGACY_INSTRUCTION,
    ):
        suffix = f"\n\n{instruction}"
        if content.endswith(suffix):
            return content[: -len(suffix)]
    return content


def build_variants(content: str) -> list[tuple[str, str, bool]]:
    """Return (name, user message, enable_thinking) diagnostics."""

    question = question_from_content(content)
    return [
        ("default_thinking_legacy_prompt", content, True),
        ("no_thinking_legacy_prompt", content, False),
        ("no_thinking_concise_boxed_prompt", f"{question}\n\n{CONCISE_INSTRUCTION}", False),
        ("no_thinking_final_only_boxed_prompt", f"{question}\n\n{FINAL_ONLY_INSTRUCTION}", False),
        ("no_thinking_short_solution_boxed_prompt", f"{question}\n\n{SHORT_SOLUTION_INSTRUCTION}", False),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument(
        "--variant",
        choices=(
            "all",
            "default_thinking_legacy_prompt",
            "no_thinking_legacy_prompt",
            "no_thinking_concise_boxed_prompt",
            "no_thinking_final_only_boxed_prompt",
            "no_thinking_short_solution_boxed_prompt",
        ),
        default="all",
        help="Prompt/template variant to measure; batch audits use the accepted short-solution contract.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if limit <= 0:
        raise ValueError("--max-prompts must be positive")
    rows = pq.read_table(path, columns=["prompt", "reward_model", "extra_info"]).to_pylist()[:limit]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}")
    if args.max_tokens <= 0 or args.samples_per_prompt <= 0:
        raise SystemExit("max-tokens and samples-per-prompt must be positive")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.utils.reward_score.math_reward import compute_score, last_boxed_only_string

    rows = load_rows(args.input, args.max_prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    records: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True)

    # This direct vLLM diagnostic uses the same TP=2 and L20 custom-all-reduce
    # fallback as the trainer, but does not allocate training GPUs or optimizer state.
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        disable_custom_all_reduce=True,
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=args.samples_per_prompt,
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    try:
        for row_index, row in enumerate(rows):
            content = str(row["prompt"][0]["content"])
            ground_truth = str(row["reward_model"]["ground_truth"])
            variants = build_variants(content)
            if args.variant != "all":
                variants = [item for item in variants if item[0] == args.variant]
            for variant, user_message, enable_thinking in variants:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_message}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                result = llm.generate([rendered], sampling, use_tqdm=False)[0]
                for sample_index, completion in enumerate(result.outputs):
                    output = completion.text
                    boxed = last_boxed_only_string(output)
                    records.append(
                        {
                            "row_index": row_index,
                            "sample_index": sample_index,
                            "variant": variant,
                            "enable_thinking": enable_thinking,
                            "ground_truth": ground_truth,
                            "response_tokens": len(completion.token_ids),
                            "hit_token_cap": len(completion.token_ids) >= args.max_tokens,
                            "finish_reason": completion.finish_reason,
                            "has_boxed_answer": boxed is not None,
                            "boxed_answer": boxed,
                            "reward": compute_score(output, ground_truth),
                            "input": rendered,
                            "output": output,
                            "extra_info": row.get("extra_info"),
                        }
                    )
    finally:
        # Explicitly release the engine before returning control to another user.
        del llm

    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_variant: dict[str, dict[str, float | int]] = {}
    for variant in sorted({record["variant"] for record in records}):
        subset = [record for record in records if record["variant"] == variant]
        by_variant[variant] = {
            "samples": len(subset),
            "boxed_answers": sum(bool(record["has_boxed_answer"]) for record in subset),
            "positive_rewards": sum(float(record["reward"]) > 0 for record in subset),
            "token_cap_hits": sum(bool(record["hit_token_cap"]) for record in subset),
            "mean_response_tokens": sum(int(record["response_tokens"]) for record in subset) / len(subset),
        }
    summary = {
        "purpose": "No-update rollout-format and reward-signal diagnostic",
        "model": str(args.model),
        "input": str(args.input),
        "max_prompts": len(rows),
        "samples_per_prompt": args.samples_per_prompt,
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "variant_selection": args.variant,
        "finish_reasons": dict(Counter(str(record["finish_reason"]) for record in records)),
        "by_variant": by_variant,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
