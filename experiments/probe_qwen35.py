"""Probe: can Qwen3.5-4B be loaded as a plain text causal LM and emit logits?"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
print("tokenizer OK, pad=", tok.pad_token_id)

model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16).to("cuda")
model.eval()
print("model class:", type(model).__name__)

inp = tok(["2+2=?"] * 2, return_tensors="pt", padding=True).to("cuda")
with torch.no_grad():
    out = model(**inp)
print("forward keys:", list(out.keys()) if hasattr(out, "keys") else type(out))
logits = out.logits if hasattr(out, "logits") else None
print("logits shape:", tuple(logits.shape) if logits is not None else "NONE")

with torch.no_grad():
    gen = model.generate(**inp, max_new_tokens=20, do_sample=False, pad_token_id=tok.pad_token_id)
print("generate OK, shape:", tuple(gen.shape))
print("decoded:", tok.decode(gen[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)[:100])
