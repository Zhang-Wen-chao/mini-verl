"""Diagnose: why does the native tool-call probe emit tools but the GRPO
rollout not? Prints the actual generated text for both paths.

Path A: probe-style, single call with tools schema (known to work 1/5).
Path B: rollout-style, same call but through the worker's _generate.
Path C: rollout-style multi-turn (turn 2 after tool response).
"""

from __future__ import annotations

import json
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else ".official-verl/models/Qwen3-0.6B"

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

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16).to("cuda")
model.eval()


def generate(messages, max_new=256):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False, tools=TOOLS)
    inp = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=True, top_p=0.9,
                             temperature=0.8, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


print("=== Path A: probe-style, 8 samples, single turn ===")
n_a = 0
for i in range(8):
    dec = generate([{"role": "user", "content": PROMPT}])
    if TOOL_CALL_RE.search(dec):
        n_a += 1
    if i == 0:
        print(f"sample 0 ({len(dec)} chars): {dec[:200]!r}")
print(f"tool calls: {n_a}/8\n")

print("=== Path B: rollout-style (exact worker kwargs), 8 samples ===")
n_b = 0
for i in range(8):
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False, tools=TOOLS,
    )
    inp = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=256, do_sample=True, top_p=0.9,
                             temperature=0.8,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    dec = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    if TOOL_CALL_RE.search(dec):
        n_b += 1
    if i == 0:
        print(f"sample 0 ({len(dec)} chars): {dec[:200]!r}")
print(f"tool calls: {n_b}/8\n")

print("=== Path C: multi-turn, turn 2 after a tool response ===")
msgs = [{"role": "user", "content": PROMPT}]
t1 = generate(msgs)
print(f"turn 1 ({len(t1)} chars): {t1[:150]!r}")
m = TOOL_CALL_RE.search(t1)
if m:
    msgs.append({"role": "assistant", "content": t1})
    msgs.append({"role": "tool", "content": "24.0"})
    t2 = generate(msgs)
    print(f"turn 2 ({len(t2)} chars): {t2[:200]!r}")
else:
    print("(no tool call in turn 1, can't build turn 2)")
