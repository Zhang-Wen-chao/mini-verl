"""Probe: does Qwen3-0.6B emit NATIVE tool-call format (<tool_call> JSON)?

Tests three variants per model:
  - thinking ON (default): model may emit <think> then <tool_call>
  - thinking OFF: enable_thinking=False, direct tool call
  - no schema: plain chat (control, no tools injected)

Counts how many of N samples emit a parseable <tool_call> block.
"""

from __future__ import annotations

import json
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else ".official-verl/models/Qwen3-0.6B"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5

PROMPT = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many "
          "in May. How many clips did Natalia sell altogether in April and May? "
          "Put your final answer after '####'.")

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

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_tool_calls(text: str) -> list[dict]:
    out = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            out.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            out.append({"unparsed": m.group(1)[:80]})
    return out


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16).to("cuda")
    model.eval()
    print(f"=== {MODEL.split('/')[-1]} (samples={N}) ===")

    variants = {
        "thinking_on": {"enable_thinking": True},
        "thinking_off": {"enable_thinking": False},
        "plain_no_tools": None,
    }

    for name, gen_kwargs in variants.items():
        n_tool = 0
        n_think = 0
        samples: list[str] = []
        for _ in range(N):
            if gen_kwargs is None:
                msgs = [{"role": "user", "content": PROMPT}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            else:
                msgs = [{"role": "user", "content": PROMPT}]
                text = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, tools=TOOLS,
                    enable_thinking=gen_kwargs["enable_thinking"],
                )
            inp = tok(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inp, max_new_tokens=256, do_sample=True, top_p=0.9,
                    temperature=0.8, pad_token_id=tok.pad_token_id,
                )
            dec = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
            calls = extract_tool_calls(dec)
            if calls:
                n_tool += 1
                samples.append((dec, calls))
            if "<think>" in dec:
                n_think += 1
        print(f"--- {name}: tool_call {n_tool}/{N}, think {n_think}/{N} ---")
        for dec, calls in samples[:1]:
            print("   calls:", json.dumps(calls, ensure_ascii=False)[:200])
            # Show the region around the first tool_call.
            m = re.search(r".{40}<tool_call>.{120}", dec, re.DOTALL)
            if m:
                print("   ctx:", m.group(0).replace("\n", " ")[:180])
        if not samples:
            print("   (no tool calls emitted)")
        print()


if __name__ == "__main__":
    main()
