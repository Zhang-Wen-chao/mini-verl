"""Multi-turn tool-calling GRPO on MATH (agent vs plain comparison-ready).

An agentic RL loop on the 2037-row OpenR1-MATH training set used by the 679-step
run: the model may call a Python tool (`[PY: <code>]`) before answering; the
tool output is appended to the conversation and the model generates again
(bounded turns). Reward is rule-based on the ground-truth answer.

Run modes:
  --tool   : agent loop with Python tool (experimental group)
  --no-tool: single-turn plain GRPO (control group, mirrors 679 run)
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import operator
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_verl.controller import Controller
from mini_verl.hf import HuggingFaceTrainerWorker, PromptExample
from mini_verl.protocol import Trajectory, TrajectoryBatch, TrajectoryValidationError
from mini_verl.workers import RuleRewardWorker

PY_RE = re.compile(r"\[PY:\s*(.*?)\]", flags=re.DOTALL)
MAX_TURNS = 3


def run_python(code: str) -> str:
    """Execute a small Python snippet in a restricted namespace and return its stdout."""
    code = code.strip()
    if len(code) > 500:
        return "error: code too long"
    # AST whitelist: no imports, no attribute access, no calls except builtins math ops.
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
               ast.Mod, ast.FloorDiv, ast.Call, ast.Load, ast.List, ast.Tuple,
               ast.Assign, ast.Expr, ast.Module, ast.Store)
    try:
        tree = ast.parse(code, mode="exec")
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                return "error: disallowed syntax"
        ns: dict = {
            "print": print,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "len": len,
            "range": range,
            "sum": sum,
            "min": min,
            "max": max,
            "pow": pow,
            "divmod": divmod,
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(tree, "<code>", "exec"), ns)
        out = buf.getvalue().strip()
        return out if out else "ok"
    except Exception as e:
        return f"error: {type(e).__name__}"


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
    """Multi-turn rollout: generate -> parse [PY: code] -> execute -> append tool
    output -> generate again, until answer or MAX_TURNS. In --no-tool mode it
    degrades to a plain single-turn rollout (control group)."""

    def __init__(self, model, tokenizer, prompts, group_size, max_new_tokens,
                 device="cuda", use_tool=True):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.use_tool = use_tool

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
        if not self.use_tool:
            return self._generate([{"role": "user", "content": prompt_text}]), []
        messages = [{"role": "user", "content": prompt_text}]
        tool_calls: list[str] = []
        last_text = ""
        for _ in range(MAX_TURNS):
            text = self._generate(messages)
            last_text = text
            calls = PY_RE.findall(text)
            if not calls:
                # No tool call requested: this turn's text is the final answer.
                return text, tool_calls
            for code in calls:
                result = run_python(code)
                tool_calls.append(code[:60])
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "tool", "content": f"PY output: {result}"})
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
    """1.0 correct answer; 0 otherwise. Tool use is not separately rewarded —
    correctness is the sole signal (same as the 679 run), so the only difference
    between groups is tool availability."""
    if trajectory.response_text is None:
        raise TrajectoryValidationError("agent_reward requires response_text")
    expected = trajectory.metadata["expected_answer"]
    answer = parse_final_answer(trajectory.response_text)
    return float(normalize(answer) == normalize(expected))


def load_math_data(parquet_path: str, limit: int | None = None) -> tuple[PromptExample, ...]:
    """Load the 2037-row OpenR1-MATH parquet (same as the 679 run) into prompts."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]
    prompts = []
    for i, row in enumerate(rows):
        # prompt is already a list of {role, content} messages.
        messages = row["prompt"]
        if isinstance(messages, str):
            text = messages
        else:
            text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        gt = row.get("reward_model", {}).get("ground_truth", "")
        prompts.append(
            PromptExample(
                text + "\nPut your final answer after '####'.",
                {
                    "group_id": f"p{i}",
                    "expected_answer": str(gt),
                },
            )
        )
    return tuple(prompts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="path to 2037-row MATH parquet")
    parser.add_argument("--limit", type=int, default=None, help="cap number of prompts")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--no-tool", action="store_true", help="control group: plain single-turn")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("agent GRPO smoke requires CUDA")
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    prompts = load_math_data(args.data, args.limit)
    print(f"loaded {len(prompts)} prompts (tool={'ON' if not args.no_tool else 'OFF'})", flush=True)

    rollout = ToolAgentRolloutWorker(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        use_tool=not args.no_tool,
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
