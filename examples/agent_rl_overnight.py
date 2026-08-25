"""Agent RL overnight experiment: tool vs no-tool, three reward versions.

Built on the existing examples/agent_grpo_smoke.py design but adds:
  - three reward versions:
      final-only : 1.0 if answer correct else 0.0          (mirrors 679 run)
      tool-bonus : final-only + 0.1 if any tool call succeeded
      process    : +0.5 if any tool call succeeded, +0.5 if answer correct
  - per-iteration tool statistics: tool_rate (fraction of trajectories that
    made a tool call), tool_success_rate, mean tool calls, mean turns
  - JSON results written to --out (one record per iteration, plus config)

Run modes:
  --mode tool    : agent loop with Python tool (experimental group)
  --mode no-tool : single-turn plain GRPO (control group, mirrors 679 run)
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import re
import time
from pathlib import Path

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
            "print": print, "abs": abs, "round": round, "int": int, "float": float,
            "str": str, "len": len, "range": range, "sum": sum, "min": min,
            "max": max, "pow": pow, "divmod": divmod,
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


def tool_was_successful(trajectory: Trajectory) -> bool:
    """A tool call counts as successful if it did not return an error."""
    return any(
        not call.startswith("error")
        for call in trajectory.metadata.get("tool_calls", [])
    )


def make_reward(reward_version: str):
    """Return a reward function for the requested reward version.

    final-only : 1.0 correct else 0.0            (tool use NOT rewarded)
    tool-bonus : final-only + 0.1 tool success   (gentle shaping)
    process    : +0.5 tool success + 0.5 correct (step-wise credit)
    """

    def reward(trajectory: Trajectory) -> float:
        if trajectory.response_text is None:
            raise TrajectoryValidationError("reward requires response_text")
        expected = trajectory.metadata["expected_answer"]
        answer = parse_final_answer(trajectory.response_text)
        correct = float(normalize(answer) == normalize(expected))
        tool_ok = float(tool_was_successful(trajectory))
        if reward_version == "final-only":
            return correct
        if reward_version == "tool-bonus":
            return correct + 0.1 * tool_ok
        if reward_version == "process":
            return 0.5 * tool_ok + 0.5 * correct
        raise ValueError(f"unknown reward version: {reward_version}")

    return reward


class ToolAgentRolloutWorker:
    """Multi-turn rollout with optional Python tool. In no-tool mode it degrades
    to a plain single-turn rollout (control group)."""

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

    def _generate_batch(self, prompt_texts: list[str]) -> list[str]:
        """Single-turn batched generation for the no-tool control group."""
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompt_texts
        ]
        inp = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inp,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return [
            self.tokenizer.decode(row[inp["input_ids"].shape[1]:], skip_special_tokens=True)
            for row in out
        ]

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
                return text, tool_calls
            for code in calls:
                result = run_python(code)
                tool_calls.append(result if result.startswith("error") else code[:60])
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "tool", "content": f"PY output: {result}"})
        return last_text, tool_calls

    def rollout(self, *, policy_version: int) -> TrajectoryBatch:
        trajectories: list[Trajectory] = []
        n_tool = 0
        n_tool_ok = 0
        n_calls = 0
        if not self.use_tool:
            # Control group: single-turn batched generation.
            all_texts = self._generate_batch([p.text for p in self.prompts for _ in range(self.group_size)])
            idx = 0
            for prompt in self.prompts:
                for i in range(self.group_size):
                    text = all_texts[idx]
                    idx += 1
                    tokens = self.tokenizer.encode(text)
                    trajectories.append(
                        Trajectory(
                            prompt_token_ids=self.tokenizer.encode(prompt.text),
                            response_token_ids=tokens,
                            old_logprobs=(0.0,) * len(tokens),
                            policy_version=policy_version,
                            group_id=prompt.metadata["group_id"],
                            response_text=text,
                            prompt_text=prompt.text,
                            metadata={**prompt.metadata, "sample_index": i, "tool_calls": []},
                        )
                    )
            self.last_tool_rate = 0.0
            self.last_tool_success_rate = 0.0
            self.last_mean_tool_calls = 0.0
            return TrajectoryBatch.from_iterable(trajectories)
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
                if tool_calls:
                    n_tool += 1
                    n_calls += len(tool_calls)
                    if any(not c.startswith("error") for c in tool_calls):
                        n_tool_ok += 1
        total = max(1, len(trajectories))
        self.last_tool_rate = n_tool / total
        self.last_tool_success_rate = n_tool_ok / max(1, n_tool)
        self.last_mean_tool_calls = n_calls / max(1, n_tool)
        return TrajectoryBatch.from_iterable(trajectories)


def load_math_data(parquet_path: str, limit: int | None = None) -> tuple[PromptExample, ...]:
    """Load the verl-format math parquet (gsm8k-smoke or OpenR1-MATH) into prompts."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]
    prompts = []
    for i, row in enumerate(rows):
        messages = row["prompt"]
        if isinstance(messages, str):
            text = messages
        else:
            text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        gt = row.get("reward_model", {}).get("ground_truth", "")
        prompts.append(
            PromptExample(
                text + "\nPut your final answer after '####'.",
                {"group_id": f"p{i}", "expected_answer": str(gt)},
            )
        )
    return tuple(prompts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="path to verl-format math parquet")
    parser.add_argument("--limit", type=int, default=None, help="cap number of prompts")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--mode", choices=["tool", "no-tool"], default="tool")
    parser.add_argument("--reward-version", choices=["final-only", "tool-bonus", "process"],
                        default="final-only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="path to output JSON")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("agent GRPO overnight requires CUDA")
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    prompts = load_math_data(args.data, args.limit)
    print(f"loaded {len(prompts)} prompts | mode={args.mode} | "
          f"reward={args.reward_version}", flush=True)

    rollout = ToolAgentRolloutWorker(
        model=model, tokenizer=tokenizer, prompts=prompts,
        group_size=args.group_size, max_new_tokens=args.max_new_tokens,
        use_tool=(args.mode == "tool"),
    )
    reward_fn = make_reward(args.reward_version)
    controller = Controller(
        rollout_worker=rollout,
        reward_worker=RuleRewardWorker(reward_fn),
        trainer_worker=HuggingFaceTrainerWorker(
            model=model, optimizer=optimizer,
            pad_token_id=int(tokenizer.pad_token_id),
            train_micro_batch_size=8,
        ),
    )

    records: list[dict] = []
    for it in range(args.iters):
        t0 = time.time()
        result = controller.run_iteration()
        dt = time.time() - t0
        tool_rate = getattr(rollout, "last_tool_rate", 0.0)
        tool_success = getattr(rollout, "last_tool_success_rate", 0.0)
        mean_calls = getattr(rollout, "last_mean_tool_calls", 0.0)
        record = {
            "iter": it,
            "mean_reward": result.mean_reward,
            "loss": result.metrics.get("loss"),
            "clip_fraction": result.metrics.get("clip_fraction"),
            "mean_response_tokens": result.metrics.get("mean_response_tokens"),
            "tool_rate": tool_rate,
            "tool_success_rate": tool_success,
            "mean_tool_calls": mean_calls,
            "seconds": round(dt, 3),
        }
        records.append(record)
        print(
            f"iter {it:3d} | reward={result.mean_reward:.3f} | loss={record['loss']:.5f} | "
            f"clip={record['clip_fraction']:.3f} | tool_rate={tool_rate:.2f} | "
            f"tool_ok={tool_success:.2f} | {dt:.1f}s", flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": vars(args),
            "records": records,
        }, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
