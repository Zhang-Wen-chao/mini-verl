"""Conservative exact-number extension for VeRL's legacy MATH reward.

The pinned scorer already owns all normal string-based cases.  This module only
adds a deliberately small fallback: two final boxed answers score equal when
both can be parsed *exactly* as an integer, finite decimal, simple fraction, or
LaTeX ``\\frac`` fraction.  It does not attempt symbolic algebra.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from typing import Callable, Optional

_INTEGER = re.compile(r"[+-]?\d+\Z")
_DECIMAL = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+)\Z")
_SLASH_FRACTION = re.compile(r"([+-]?\d+)\s*/\s*([+-]?\d+)\Z")
_LATEX_FRACTION = re.compile(r"([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}\Z")


def last_boxed_answer(text: str) -> Optional[str]:
    """Return the last balanced ``\boxed{...}`` / ``\fbox{...}`` body."""
    start = max(text.rfind(r"\boxed{"), text.rfind(r"\fbox{"))
    if start < 0:
        return None
    body_start = text.find("{", start) + 1
    depth = 1
    for index in range(body_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[body_start:index]
    return None


def _strip_outer_braces(value: str) -> str:
    while value.startswith("{") and value.endswith("}"):
        depth = 0
        closes_at_end = False
        for index, char in enumerate(value):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    closes_at_end = index == len(value) - 1
                    break
        if not closes_at_end:
            break
        value = value[1:-1]
    return value


def exact_rational(value: str) -> Optional[Fraction]:
    """Parse only an unambiguous, finite rational answer; otherwise ``None``."""
    cleaned = value.strip().replace(r"\left", "").replace(r"\right", "")
    cleaned = cleaned.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    cleaned = cleaned.replace("$", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = _strip_outer_braces(cleaned)
    if _INTEGER.fullmatch(cleaned):
        # A leading-zero answer can encode a composite answer, such as 09.
        # Keep that ambiguous notation under the legacy scorer.
        unsigned = cleaned.lstrip("+-")
        if len(unsigned) > 1 and unsigned.startswith("0"): 
            return None
        return Fraction(int(cleaned), 1)
    if _DECIMAL.fullmatch(cleaned):
        try:
            return Fraction(Decimal(cleaned))
        except InvalidOperation:
            return None
    slash = _SLASH_FRACTION.fullmatch(cleaned)
    if slash:
        numerator, denominator = map(int, slash.groups())
        return None if denominator == 0 else Fraction(numerator, denominator)
    latex = _LATEX_FRACTION.fullmatch(cleaned)
    if latex:
        sign, numerator, denominator = latex.groups()
        numerator_int, denominator_int = int(numerator), int(denominator)
        if denominator_int == 0:
            return None
        return Fraction(-numerator_int if sign == "-" else numerator_int, denominator_int)
    return None


def compute_score(solution_str: str, ground_truth: str, legacy_compute_score: Callable[[str, str], float]) -> float:
    """Preserve legacy scoring, then allow exact rational equivalence only."""
    legacy_score = float(legacy_compute_score(solution_str, ground_truth))
    if legacy_score:
        return legacy_score
    answer = last_boxed_answer(solution_str)
    if answer is None:
        return legacy_score
    predicted = exact_rational(answer)
    expected = exact_rational(ground_truth)
    return 1.0 if predicted is not None and expected is not None and predicted == expected else legacy_score
