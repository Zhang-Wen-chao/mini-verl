"""Probe: what does Qwen3.5-4B actually output with tools schema? Show raw text."""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]
PROMPT = ("What is 17*23+45? Put your final answer after '####'.")
TOOLS = [{
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run Python code and return its stdout.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
}]

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16).to("cuda")
model.eval()

for thinking in (False, True):
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True,
        tools=TOOLS, enable_thinking=thinking,
    )
    inp = tok(text, return_tensors="pt").to("cuda")
    for attempt in range(3):
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=300, do_sample=True, top_p=0.9,
                                 temperature=0.8, pad_token_id=tok.pad_token_id)
        dec = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"--- thinking={thinking} attempt={attempt} ({len(dec)} chars) ---")
        print(repr(dec[:400]))
        print()
