"""Qwen NATIVE tool-call format GRPO: does GRPO make the model USE the tool?

Follow-up to the overnight report's finding that Qwen3-0.6B never emits the
custom `[PY: ...]` format. This script uses the model's OWN tool protocol:
  - thinking OFF (enable_thinking=False) so it emits direct tool calls
  - tools schema injected via apply_chat_template(tools=[...])
  - parse <tool_call>{"name": "python", "arguments": {"code": ...}}</tool_call>
  - execute in the AST sandbox, append result as <tool_response>...</tool_response>
  - multi-turn loop (MAX_TURNS), reward = final answer correctness

Run modes:
  --mode tool    : native tool loop (experimental)
  --mode no-tool : plain single-turn, no tools schema (control)

Writes per-iteration JSON (reward / tool_rate / tool_success / clip / loss).
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

MAX_TURNS = 3
# Two native Qwen tool-call formats:
#   JSON:      <tool_call>{"name": "python", "arguments": {"code": "..."}}</tool_call>   (Qwen3-0.6B)
#   function:  <tool_call><function=python><parameter=code>...</parameter></function></tool_call>  (Qwen3.5-4B)
JSON_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
FUNC_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>\s*<parameter=(\w+)>\s*(.*?)\s*</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL,
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run Python code and return its stdout. Use for arithmetic.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code"}},
            "required": ["code"],
        },
    },
}]


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


def extract_tool_calls(text: str) -> list[dict]:
    """Extract tool calls in either native format. Returns [{name, code}]."""
    out: list[dict] = []
    for m in JSON_TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                args = obj.get("arguments", {})
                out.append({"name": obj.get("name", ""), "code": args.get("code", "")})
        except json.JSONDecodeError:
            pass
    for m in FUNC_TOOL_CALL_RE.finditer(text):
        out.append({"name": m.group(1), "code": m.group(3).strip()})
    return out


class QwenToolRolloutWorker:
    """Multi-turn rollout using Qwen's native tool-call protocol (thinking off)."""

    def __init__(self, model, tokenizer, prompts, group_size, max_new_tokens,
                 device="cuda", use_tool=True, enable_thinking=False):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.use_tool = use_tool
        self.enable_thinking = enable_thinking

    def _generate(self, messages, *, tools: bool) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True,
                  "enable_thinking": self.enable_thinking}
        if tools:
            kwargs["tools"] = TOOLS
        text = self.tokenizer.apply_chat_template(messages, **kwargs)
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
            return self._generate([{"role": "user", "content": prompt_text}], tools=False), []
        messages = [{"role": "user", "content": prompt_text}]
        tool_calls: list[str] = []
        last_text = ""
        for _ in range(MAX_TURNS):
            text = self._generate(messages, tools=True)
            last_text = text
            calls = extract_tool_calls(text)
            if not calls:
                # No tool call: this turn's text is the final answer.
                return text, tool_calls
            messages.append({"role": "assistant", "content": text})
            for call in calls:
                code = call.get("code", "")
                result = run_python(code)
                status = "ok" if not result.startswith("error") else "error"
                tool_calls.append(f"{status}:{code[:40]}")
                messages.append({"role": "tool", "content": result})
        return last_text, tool_calls

    def rollout(self, *, policy_version: int) -> TrajectoryBatch:
        trajectories: list[Trajectory] = []
        n_tool = 0
        n_tool_ok = 0
        n_calls = 0
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
                    if any(c.startswith("ok:") for c in tool_calls):
                        n_tool_ok += 1
        total = max(1, len(trajectories))
        self.last_tool_rate = n_tool / total
        self.last_tool_success_rate = n_tool_ok / max(1, n_tool)
        self.last_mean_tool_calls = n_calls / max(1, n_tool)
        return TrajectoryBatch.from_iterable(trajectories)


def exact_answer_reward(trajectory: Trajectory) -> float:
    if trajectory.response_text is None:
        raise TrajectoryValidationError("reward requires response_text")
    expected = trajectory.metadata["expected_answer"]
    answer = parse_final_answer(trajectory.response_text)
    return float(normalize(answer) == normalize(expected))


def load_math_data(parquet_path: str, limit: int | None = None) -> tuple[PromptExample, ...]:
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
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--train-micro-batch", type=int, default=8,
                        help="trainer micro-batch size (lower for big models)")
    parser.add_argument("--mode", choices=["tool", "no-tool"], default="tool")
    parser.add_argument("--thinking", action="store_true",
                        help="enable thinking mode (Qwen3.5-4B needs it for tools)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")
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
    print(f"loaded {len(prompts)} prompts | mode={args.mode} | model={args.model}", flush=True)

    rollout = QwenToolRolloutWorker(
        model=model, tokenizer=tokenizer, prompts=prompts,
        group_size=args.group_size, max_new_tokens=args.max_new_tokens,
        use_tool=(args.mode == "tool"),
        enable_thinking=args.thinking,
    )
    controller = Controller(
        rollout_worker=rollout,
        reward_worker=RuleRewardWorker(exact_answer_reward),
        trainer_worker=HuggingFaceTrainerWorker(
            model=model, optimizer=optimizer,
            pad_token_id=int(tokenizer.pad_token_id),
            train_micro_batch_size=args.train_micro_batch,
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
        json.dump({"config": vars(args), "records": records}, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
