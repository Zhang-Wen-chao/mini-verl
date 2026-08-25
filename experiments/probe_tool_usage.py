"""Probe: does Qwen3-0.6B ever emit [PY: ...] tool calls, with/without guidance?"""
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else ".official-verl/models/Qwen3-0.6B"
PROMPT = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many "
          "in May. How many clips did Natalia sell altogether in April and May? "
          "Put your final answer after '####'.")

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16).to("cuda")
model.eval()

variants = {
    "plain": [{"role": "user", "content": PROMPT}],
    "guided": [
        {"role": "system", "content": "You have access to a Python tool. When you need to compute something, "
         "write [PY: <python code>] and you will see the tool output before answering."},
        {"role": "user", "content": PROMPT},
    ],
    "guided_short": [
        {"role": "system", "content": "Use the Python tool [PY: <code>] to calculate. Then give the final answer after ####."},
        {"role": "user", "content": PROMPT},
    ],
}

for name, msgs in variants.items():
    n_py = 0
    samples = []
    for _ in range(5):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=160, do_sample=True, top_p=0.9,
                                 temperature=0.8, pad_token_id=tok.pad_token_id)
        dec = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        if re.search(r"\[PY:", dec):
            n_py += 1
            samples.append(dec[:200])
    print(f"=== {name}: {n_py}/5 emitted [PY: ...] ===")
    for s in samples[:2]:
        print("  ", s.replace("\n", " ")[:180])
    print()
