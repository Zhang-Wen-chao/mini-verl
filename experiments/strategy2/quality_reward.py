"""Pure helpers for the quality-aware Strategy 2 reward.

The helpers intentionally have no Slime or sandbox dependency so the reward
policy can be unit-tested without starting a rollout worker.
"""

from __future__ import annotations

import re


_ERROR_OBSERVATION = re.compile(
    r"^\s*(?:error|errors|traceback|exception|syntaxerror|importerror|nameerror)\b",
    re.IGNORECASE,
)
_MARKDOWN_CODE_FENCE = re.compile(r"^\s*```(?:python|py)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def tool_execution_succeeded(observation: str) -> bool:
    """Return whether the sandbox executed the submitted program.

    An empty observation is a valid successful execution: many useful Python
    snippets compute a value without printing it. Sandbox rejections, runtime
    errors, and timeouts all have an error-like leading marker.
    """
    return not bool(_ERROR_OBSERVATION.search(observation))


def code_fingerprint(code: str) -> str:
    """Normalize insignificant whitespace for repeated-tool-call detection."""
    return re.sub(r"\s+", " ", code).strip()


def normalize_markdown_code(code: str) -> tuple[str, bool]:
    """Remove only an outer Markdown code fence from a tool JSON field.

    The tool contract is raw Python, but the policy frequently emits standard
    `````py`` fences.  Normalizing this recoverable protocol drift before the
    AST safety check makes training and held-out evaluation use the same tool
    semantics.  The boolean lets reward and metrics distinguish it from clean
    tool syntax; no content inside the fence is changed.
    """
    match = _MARKDOWN_CODE_FENCE.match(code)
    return (match.group(1), True) if match else (code, False)


def quality_process_score(
    outcome_score: float,
    *,
    submitted_answer: bool,
    tool_successes: int,
    tool_failures: int,
    invalid_actions: int,
    repeated_tool_calls: int,
    markdown_fenced_tool_calls: int = 0,
) -> float:
    """Blend outcome reward with bounded, auditable process signals.

    Correct answers remain dominant. Successful tool execution earns only a
    small partial credit; malformed actions, sandbox or runtime errors,
    repeated identical calls, Markdown repair, and failing to submit an answer
    all reduce reward--including when the final answer happens to be correct.
    Incorrect trajectories are capped below zero so tool activity alone cannot
    outrank a verified answer.
    """
    successes = min(max(tool_successes, 0), 2)
    failures = max(tool_failures, 0)
    invalid = max(invalid_actions, 0)
    repeats = max(repeated_tool_calls, 0)
    fences = max(markdown_fenced_tool_calls, 0)

    success_credit = 0.06 * successes
    completion_credit = 0.04 if submitted_answer and successes else 0.0
    penalties = 0.08 * failures + 0.05 * invalid + 0.05 * repeats + 0.02 * fences
    if not submitted_answer:
        penalties += 0.08

    if outcome_score > 0:
        # Do not forgive a noisy path merely because it eventually reached the
        # correct answer.  This preserves the outcome as the dominant signal,
        # while teaching the policy to use a clean, efficient tool trajectory.
        return outcome_score + success_credit - penalties

    return min(-0.20, -1.0 + success_credit + completion_credit - penalties)
