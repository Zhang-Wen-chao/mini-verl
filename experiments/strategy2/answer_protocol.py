"""Final-answer extraction shared by Strategy 2 training and evaluation.

The rollout protocol permits tool code containing comments such as ``Answer:``.
It also appends recovery messages after malformed turns.  Neither is model
output that should be scored as a mathematical final answer.  Keeping this
logic in one dependency-free module makes those boundaries testable.
"""

from __future__ import annotations

import re


_BOXED_ANSWER_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_EXPLICIT_ANSWER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:final\s+)?answer\s*:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_CALL_BLOCK_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_CODE_TAG_BLOCK_PATTERN = re.compile(r"<code>.*?</code>", re.DOTALL)
_INTERPRETER_BLOCK_PATTERN = re.compile(r"<interpreter>.*?</interpreter>", re.DOTALL)
_FENCED_CODE_BLOCK_PATTERN = re.compile(r"```(?:python|py)?\s*.*?```", re.IGNORECASE | re.DOTALL)
_PLACEHOLDER_ANSWERS = {"answer", "<answer>", "your answer", "...", "<value>", "value"}


def _visible_text(text: str) -> str:
    """Remove action payloads before looking for a natural-language answer."""

    text = _TOOL_CALL_BLOCK_PATTERN.sub("", text)
    text = _CODE_TAG_BLOCK_PATTERN.sub("", text)
    text = _INTERPRETER_BLOCK_PATTERN.sub("", text)
    return _FENCED_CODE_BLOCK_PATTERN.sub("", text)


def _is_placeholder(value: str) -> bool:
    return " ".join(value.split()).casefold() in _PLACEHOLDER_ANSWERS


def extract_final_answer(text: str) -> str | None:
    """Return the last real boxed answer emitted outside tool/code payloads.

    An explicit ``Answer:`` is preferred, but a bare final ``\\boxed{...}``
    is accepted for compatibility with common math-model formatting.  Values
    such as ``\\boxed{answer}``, which historically appeared in recovery
    prompts, are deliberately ignored.
    """

    visible = _visible_text(text)
    explicit = [match.group(1).strip() for match in _EXPLICIT_ANSWER_PATTERN.finditer(visible)]
    for value in reversed(explicit):
        if value and not _is_placeholder(value):
            return value

    boxed = [match.group(1).strip() for match in _BOXED_ANSWER_PATTERN.finditer(visible)]
    for value in reversed(boxed):
        if value and not _is_placeholder(value):
            return value
    return None


def scoreable_answer_text(answer: str | None) -> str:
    """Render only a verified extracted answer for Math-DAPO scoring."""

    if not answer:
        return ""
    return f"Answer: \\boxed{{{answer}}}"
