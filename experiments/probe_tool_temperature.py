"""Probe: does lower temperature / different sampling increase native
tool-call rate on Qwen3-0.6B (thinking off, tools schema)?"""

from __future__ import annotations

import json
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else ".official-verl/models/Qwen3-0.6B"
N = 10

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

TOOL_CALL_RE = re.compile(r"<tool_call>", re.DOTALL)

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16).to("cuda")
model.eval()

variants = [
    ("temp0.8_top0.9", 0.8, 0.9),
    ("temp0.6_top0.95", 0.6, 0.95),
    ("temp0.4_top0.95", 0.4, 0.95),
    ("temp1.0_top0.9", 1.0, 0.9),
]

msgs = [{"role": "user", "content": PROMPT}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                               enable_thinking=False, tools=TOOLS)
inp = tok(text, return_tensors="pt").to("cuda")

for name, temp, top_p in variants:
    n_tool = 0
    n_ok = 0
    for _ in range(N):
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, do_sample=True, top_p=top_p,
                                 temperature=temp, pad_token_id=tok.pad_token_id)
        dec = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        if TOOL_CALL_RE.search(dec):
            n_tool += 1
            if "####" in dec:
                n_ok += 1
    print(f"{name}: tool_call {n_tool}/{N}, with_answer {n_ok}/{N}")
