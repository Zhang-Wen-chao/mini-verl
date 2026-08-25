"""Probe: how many tokens does Qwen3-0.6B need to finish (#### answer) on gsm8k?"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]
import pyarrow.parquet as pq

DATA = sys.argv[2] if len(sys.argv) > 2 else ".official-verl/data/gsm8k-smoke/train.parquet"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 4

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16).to("cuda")
model.eval()

table = pq.read_table(DATA).to_pylist()[:LIMIT]
print(f"probing {len(table)} prompts...")
for i, row in enumerate(table):
    msgs = row["prompt"]
    text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs)
    text += "\nPut your final answer after '####'."
    full = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    inp = tok(full, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=512, do_sample=True, top_p=0.9,
                             temperature=0.8, pad_token_id=tok.pad_token_id)
    dec = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    n_tokens = out.shape[1] - inp["input_ids"].shape[1]
    has_answer = "####" in dec
    print(f"  prompt {i}: {n_tokens} tokens generated, has '####': {has_answer}")
    if has_answer:
        idx = dec.index("####")
        print(f"    ...answer context: ...{dec[max(0,idx-30):idx+40]!r}")
