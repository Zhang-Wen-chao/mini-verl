"""Run one local Hugging Face CausalLM rollout/reward/GRPO update.

This is a backend-plumbing smoke, not a quality evaluation. It uses a
deterministic per-sample reward only to make the GRPO advantage non-degenerate.
Pass a complete local model snapshot; no network download is attempted.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_verl.controller import Controller
from mini_verl.hf import HuggingFaceRolloutWorker, HuggingFaceTrainerWorker, PromptExample
from mini_verl.protocol import Trajectory
from mini_verl.workers import RuleRewardWorker


def smoke_reward(trajectory: Trajectory) -> float:
    """Produce two reward values per prompt group without judging text quality."""
    return float(trajectory.metadata["sample_index"] == 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Complete local Hugging Face model snapshot")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this CausalLM smoke requires CUDA")
    torch.manual_seed(11)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    controller = Controller(
        rollout_worker=HuggingFaceRolloutWorker(
            model=model,
            tokenizer=tokenizer,
            prompts=(
                PromptExample("Reply with a one-sentence greeting.", {"task": "greeting"}),
                PromptExample("What is 2 + 2? Answer briefly.", {"task": "arithmetic"}),
            ),
            group_size=2,
            max_new_tokens=args.max_new_tokens,
        ),
        reward_worker=RuleRewardWorker(smoke_reward),
        trainer_worker=HuggingFaceTrainerWorker(
            model=model,
            optimizer=optimizer,
            pad_token_id=int(tokenizer.pad_token_id),
        ),
    )
    result = controller.run_iteration()
    print(f"policy_version={result.policy_version}->{result.next_policy_version}")
    print(f"trajectories={result.trajectory_count} response_tokens={result.response_token_count}")
    print(f"mean_reward={result.mean_reward:.3f} loss={result.metrics['loss']:.6f}")
    print(f"mean_ratio={result.metrics['mean_ratio']:.6f} clip_fraction={result.metrics['clip_fraction']:.6f}")


if __name__ == "__main__":
    main()
