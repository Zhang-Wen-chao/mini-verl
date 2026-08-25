"""Probe: Qwen3 multi-turn tool-call round trip with native format.

Shows how the chat template renders: user -> assistant tool_call ->
tool result -> assistant final. Prints the template text so we can confirm
the message format the rollout worker must produce.
"""

from __future__ import annotations

import sys

from transformers import AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else ".official-verl/models/Qwen3-0.6B"

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run Python code and return stdout.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
}]

# Turn 1: render with tools.
msgs1 = [{"role": "user", "content": "What is 48 * 0.5? Put final answer after ####."}]
t1 = tok.apply_chat_template(msgs1, tokenize=False, add_generation_prompt=True, tools=TOOLS,
                             enable_thinking=False)
print("=== TURN 1 (with tools, thinking off) ===")
print(t1[:1200])
print()

# Simulated model output (tool call), then tool result appended.
msgs2 = msgs1 + [
    {"role": "assistant", "content": '<tool_call>{"name": "python", "arguments": {"code": "print(48 * 0.5)"}}</tool_call>'},
    {"role": "tool", "content": "24.0"},
]
t2 = tok.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True, tools=TOOLS,
                             enable_thinking=False)
print("=== TURN 2 (after tool result, thinking off) ===")
print(t2[-800:])
