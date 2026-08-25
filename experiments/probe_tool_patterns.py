"""Probe: what do Qwen models actually emit for tool-ish content? Search for tool-like tokens."""
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else ".official-verl/models/Qwen3-0.6B"
PROMPT = ("What is 17*23+45? Show your work. "
          "Put your final answer after '####'.")

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16).to("cuda")
model.eval()

patterns = {
    "PY": r"\[PY:",
    "tool_call": r"tool_call",
    "python": r"python",
    "calc": r"calc",
    "think": r"<think>|/think|思考",
    "code_block": r"```",
    "boxed": r"\\boxed",
    "backtick": r"`",
}

msgs = [
    {"role": "system", "content": "You have a Python tool. Use [PY: code] to compute."},
    {"role": "user", "content": PROMPT},
]
hits = {k: 0 for k in patterns}
samples = []
for _ in range(4):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=180, do_sample=True, top_p=0.9,
                             temperature=0.8, pad_token_id=tok.pad_token_id)
    dec = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    samples.append(dec)
    for k, p in patterns.items():
        if re.search(p, dec, re.IGNORECASE):
            hits[k] += 1

print(f"=== {MODEL.split('/')[-1]} pattern hits (of 4 samples) ===")
for k, v in hits.items():
    print(f"  {k}: {v}/4")
print("\n--- first sample (600 chars) ---")
print(samples[0][:600] if samples else "none")
