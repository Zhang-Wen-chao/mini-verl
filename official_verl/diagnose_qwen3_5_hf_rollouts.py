"""Run a no-update Qwen3.5 math format diagnostic with Hugging Face generation.

This is the fallback for the container where a standalone vLLM engine cannot
obtain its shared-memory broadcast block.  It purposefully avoids Ray, vLLM,
FSDP, optimizers, gradients, and checkpoints.  It tests only whether Qwen's
native chat template and prompt produce parsable final answers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any



def _shared_diagnostic_module():
    """Load the adjacent shared prompt/data contract in script and test modes."""

    path = Path(__file__).with_name("diagnose_qwen3_5_rollouts.py")
    spec = importlib.util.spec_from_file_location("official_verl_rollout_diagnostic_shared", path)
    if spec is None or spec.loader is None:  # pragma: no cover - regular file invariant
        raise RuntimeError(f"could not load shared diagnostic helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_shared = _shared_diagnostic_module()
build_variants = _shared.build_variants
load_rows = _shared.load_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument(
        "--max-tokens",
        type=int,
        nargs="+",
        default=[512],
        help="One or more completion lengths to sweep without reloading the model.",
    )
    parser.add_argument(
        "--variant",
        choices=(
            "all",
            "no_thinking_concise_boxed_prompt",
            "no_thinking_final_only_boxed_prompt",
            "no_thinking_short_solution_boxed_prompt",
        ),
        default="all",
        help=(
            "Prompt/template variant to measure.  The concise-only mode is for "
            "response-length sweeps after the format comparison has identified it "
            "as the candidate training contract."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def finish_reason(new_token_count: int, max_tokens: int, eos_token_id: int | None, token_ids: list[int]) -> str:
    if new_token_count >= max_tokens:
        return "length"
    if eos_token_id is not None and token_ids and token_ids[-1] == eos_token_id:
        return "stop"
    return "unknown"


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}")
    if args.max_prompts <= 0 or args.samples_per_prompt <= 0 or any(tokens <= 0 for tokens in args.max_tokens):
        raise SystemExit("prompt count, samples, and max tokens must be positive")

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
    from verl.utils.reward_score.math_reward import compute_score, last_boxed_only_string

    rows = load_rows(args.input, args.max_prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    records: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True)

    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as sample_handle:
        with torch.inference_mode():
            for row_index, row in enumerate(rows):
                content = str(row["prompt"][0]["content"])
                ground_truth = str(row["reward_model"]["ground_truth"])
                variants = build_variants(content)
                if args.variant != "all":
                    variants = [item for item in variants if item[0] == args.variant]
                for variant_index, (variant, user_message, enable_thinking) in enumerate(variants):
                    rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_message}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                    batch = tokenizer(rendered, return_tensors="pt").to("cuda")
                    # A deterministic per-prompt seed makes this small diagnostic reproducible.
                    set_seed(args.seed + row_index * 10 + variant_index)
                    prompt_tokens = int(batch["input_ids"].shape[1])
                    for max_tokens in args.max_tokens:
                        generated = model.generate(
                        **batch,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        max_new_tokens=max_tokens,
                        num_return_sequences=args.samples_per_prompt,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                        for sample_index, token_ids_tensor in enumerate(generated):
                            token_ids = token_ids_tensor[prompt_tokens:].tolist()
                            output = tokenizer.decode(token_ids, skip_special_tokens=True)
                            boxed = last_boxed_only_string(output)
                            record = (
                            {
                                "row_index": row_index,
                                "sample_index": sample_index,
                                "variant": variant,
                                "enable_thinking": enable_thinking,
                                "max_tokens": max_tokens,
                                "ground_truth": ground_truth,
                                "response_tokens": len(token_ids),
                                "hit_token_cap": len(token_ids) >= max_tokens,
                                "finish_reason": finish_reason(
                                    len(token_ids), max_tokens, tokenizer.eos_token_id, token_ids
                                ),
                                "has_boxed_answer": boxed is not None,
                                "boxed_answer": boxed,
                                "reward": compute_score(output, ground_truth),
                                "input": rendered,
                                "output": output,
                                "extra_info": row.get("extra_info"),
                            }
                            )
                            records.append(record)
                            sample_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        sample_handle.flush()
                        print(
                            f"completed row={row_index} variant={variant} max_tokens={max_tokens} "
                            f"records={len(records)}",
                            flush=True,
                        )

    by_max_tokens: dict[str, dict[str, float | int]] = {}
    for max_tokens in sorted({int(record["max_tokens"]) for record in records}):
        subset = [record for record in records if int(record["max_tokens"]) == max_tokens]
        by_max_tokens[str(max_tokens)] = {
            "samples": len(subset),
            "boxed_answers": sum(bool(record["has_boxed_answer"]) for record in subset),
            "positive_rewards": sum(float(record["reward"]) > 0 for record in subset),
            "token_cap_hits": sum(bool(record["hit_token_cap"]) for record in subset),
            "mean_response_tokens": sum(int(record["response_tokens"]) for record in subset) / len(subset),
        }
    summary = {
        "purpose": "No-update Hugging Face rollout-format and reward-signal diagnostic",
        "model": str(args.model),
        "input": str(args.input),
        "max_prompts": len(rows),
        "samples_per_prompt": args.samples_per_prompt,
        "max_tokens": args.max_tokens,
        "variant_selection": args.variant,
        "backend": "transformers_hf_single_gpu",
        "finish_reasons": dict(Counter(str(record["finish_reason"]) for record in records)),
        "by_max_tokens": by_max_tokens,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
