"""Probe: Qwen3.5-4B native tool protocol details.

Shows: (1) the tools-system prompt rendered by apply_chat_template,
(2) the exact tool_call format the model emits, (3) how a tool result
should be appended for turn 2 (role name + content wrapper).
"""

from __future__ import annotations

import sys

from transformers import AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else ".official-verl/models/Qwen3.5-4B"

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run Python code and return stdout.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
}]

# Turn 1 render
msgs1 = [{"role": "user", "content": "What is 48*0.5? Put final answer after ####."}]
t1 = tok.apply_chat_template(msgs1, tokenize=False, add_generation_prompt=True, tools=TOOLS,
                             enable_thinking=True)
print("=== TURN 1 (thinking on, tools) ===")
print(t1[:1500])
print()

# Try tool result as role=tool
msgs2 = msgs1 + [
    {"role": "assistant", "content": '<tool_call>\n<function=python>\n<parameter=code>\nprint(48*0.5)\n</parameter>\n</function>\n</tool_call>'},
    {"role": "tool", "content": "24.0"},
]
t2 = tok.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True, tools=TOOLS,
                             enable_thinking=True)
print("=== TURN 2 (role=tool) tail ===")
print(t2[-600:])
