"""Multi-turn tool-calling GRPO smoke on mini-verl.

A minimal agentic RL loop: the model may call a calculator tool
(`[CALC: <expr>]`) before answering; the tool result is appended to the
conversation and the model generates again (bounded turns). Reward is
rule-based: 1.0 for a correct final answer AND at least one tool call,
0.5 for a correct answer without a tool call, 0 otherwise. This trains
the model to *use the tool* to solve arithmetic problems.

Uses the same Controller/GRPO machinery as hf_grpo_smoke.py; the only new
piece is ToolAgentRolloutWorker which implements the multi-turn loop.
"""

from __future__ import annotations

import argparse
import ast
import operator
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_verl.controller import Controller
from mini_verl.hf import HuggingFaceTrainerWorker, PromptExample
from mini_verl.protocol import Trajectory, TrajectoryBatch, TrajectoryValidationError
from mini_verl.workers import RuleRewardWorker

CALC_RE = re.compile(r"\[CALC:\s*([^\]]+)\]")
MAX_TURNS = 3


def safe_eval(expr: str) -> str:
    """Evaluate a simple arithmetic expression safely (no eval of arbitrary code)."""
    expr = expr.strip()
    tree = ast.parse(expr, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            return "error"
    try:
        # compile + restricted globals still allows operators via bytecode; the
        # AST whitelist above is the real gate. literal_eval-based evaluation
        # would not support operators, so use compile with empty builtins.
        return str(eval(compile(tree, "<string>", "eval"), {"__builtins__": {}}))
    except Exception:
        return "error"


def parse_final_answer(text: str) -> str:
    """Extract the answer after '####' or the last number in the response."""
    m = re.search(r"####\s*(-?[\d.,]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1].strip() if nums else ""


def normalize(n: str) -> str:
    try:
        return str(float(n))
    except ValueError:
        return n.strip()


class ToolAgentRolloutWorker:
    """Multi-turn tool-calling rollout: generate -> parse [CALC:] -> execute ->
    append tool result to messages -> generate again, until answer or MAX_TURNS.
    """

    def __init__(self, model, tokenizer, prompts, group_size, max_new_tokens, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.device = device

    def _generate(self, messages) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inp = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inp,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def _run_episode(self, prompt_text: str) -> tuple[str, list[str]]:
        """Run one multi-turn episode. Returns (final_text, tool_calls)."""
        messages = [{"role": "user", "content": prompt_text}]
        tool_calls: list[str] = []
        last_text = ""
        for _ in range(MAX_TURNS):
            text = self._generate(messages)
            last_text = text
            calls = CALC_RE.findall(text)
            if not calls:
                # No tool call requested: this turn's text is the final answer.
                return text, tool_calls
            for expr in calls:
                result = safe_eval(expr)
                tool_calls.append(expr)
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "tool", "content": f"CALC {expr} = {result}"})
        # Exhausted turns: return the last generated text.
        return last_text, tool_calls

    def rollout(self, *, policy_version: int) -> TrajectoryBatch:
        trajectories: list[Trajectory] = []
        for prompt in self.prompts:
            for i in range(self.group_size):
                text, tool_calls = self._run_episode(prompt.text)
                tokens = self.tokenizer.encode(text)
                trajectories.append(
                    Trajectory(
                        prompt_token_ids=self.tokenizer.encode(prompt.text),
                        response_token_ids=tokens,
                        old_logprobs=(0.0,) * len(tokens),  # placeholder; replaced by trainer
                        policy_version=policy_version,
                        group_id=prompt.metadata["group_id"],
                        response_text=text,
                        prompt_text=prompt.text,
                        metadata={
                            **prompt.metadata,
                            "sample_index": i,
                            "tool_calls": tool_calls,
                        },
                    )
                )
        return TrajectoryBatch.from_iterable(trajectories)


def agent_reward(trajectory: Trajectory) -> float:
    """1.0 correct answer + used tool; 0.5 correct without tool; 0 otherwise."""
    if trajectory.response_text is None:
        raise TrajectoryValidationError("agent_reward requires response_text")
    expected = trajectory.metadata["expected_answer"]
    answer = parse_final_answer(trajectory.response_text)
    correct = normalize(answer) == normalize(expected)
    used_tool = len(trajectory.metadata.get("tool_calls", [])) > 0
    if correct and used_tool:
        return 1.0
    if correct:
        return 0.5
    return 0.0


PROBLEMS = [
    ("Compute 17 * 23 and answer with the number after '####'.", "391"),
    ("Compute 144 / 12 and answer with the number after '####'.", "12"),
    ("Compute (5 + 7) * 3 and answer with the number after '####'.", "36"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("agent GRPO smoke requires CUDA")
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    prompts = tuple(
        PromptExample(
            text,
            {
                "group_id": f"p{i}",
                "expected_answer": ans,
            },
        )
        for i, (text, ans) in enumerate(PROBLEMS)
    )

    rollout = ToolAgentRolloutWorker(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
    )
    controller = Controller(
        rollout_worker=rollout,
        reward_worker=RuleRewardWorker(agent_reward),
        trainer_worker=HuggingFaceTrainerWorker(
            model=model,
            optimizer=optimizer,
            pad_token_id=int(tokenizer.pad_token_id),
        ),
    )

    for it in range(args.iters):
        result = controller.run_iteration()
        print(
            f"iter {it:3d} | reward={result.mean_reward:.3f} | "
            f"loss={result.metrics['loss']:.5f} | "
            f"trajectories={result.trajectory_count} | "
            f"clip={result.metrics['clip_fraction']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
