"""GRPO's masked token-level objective.

This module has two implementations of the same loss. `grpo_loss_reference` is
dependency-free and deliberately scalar, which makes its numerical behavior easy
to test. `torch_grpo_loss` is the differentiable implementation used by a trainer
once PyTorch is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class GrpoLossTerms:
    """Scalar diagnostics from one masked GRPO loss calculation."""

    total_loss: float
    policy_loss: float
    kl_loss: float
    mean_ratio: float
    clip_fraction: float
    token_count: int


def _as_rows(name: str, rows: Iterable[Iterable[float]]) -> tuple[tuple[float, ...], ...]:
    result = tuple(tuple(float(value) for value in row) for row in rows)
    if not result or not result[0]:
        raise ValueError(f"{name} must be a non-empty rectangular matrix")
    width = len(result[0])
    if any(len(row) != width for row in result):
        raise ValueError(f"{name} must be rectangular")
    if any(not math.isfinite(value) for row in result for value in row):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _as_mask(mask: Iterable[Iterable[bool]], expected_shape: tuple[int, int]) -> tuple[tuple[bool, ...], ...]:
    result = tuple(tuple(row) for row in mask)
    if len(result) != expected_shape[0] or any(len(row) != expected_shape[1] for row in result):
        raise ValueError("response_mask must have the same shape as logprob matrices")
    if any(not isinstance(value, bool) for row in result for value in row):
        raise ValueError("response_mask must only contain booleans")
    return result


def _validate_inputs(
    new_logprobs: Iterable[Iterable[float]],
    old_logprobs: Iterable[Iterable[float]],
    reference_logprobs: Iterable[Iterable[float]] | None,
    advantages: Iterable[float],
    response_mask: Iterable[Iterable[bool]],
    clip_range: float,
    beta: float,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...] | None,
    tuple[float, ...],
    tuple[tuple[bool, ...], ...],
]:
    if clip_range < 0:
        raise ValueError("clip_range must be non-negative")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    new = _as_rows("new_logprobs", new_logprobs)
    old = _as_rows("old_logprobs", old_logprobs)
    if (len(old), len(old[0])) != (len(new), len(new[0])):
        raise ValueError("old_logprobs must match new_logprobs shape")
    reference = None if reference_logprobs is None else _as_rows("reference_logprobs", reference_logprobs)
    if reference is not None and (len(reference), len(reference[0])) != (len(new), len(new[0])):
        raise ValueError("reference_logprobs must match new_logprobs shape")
    advantage_values = tuple(float(value) for value in advantages)
    if len(advantage_values) != len(new) or any(not math.isfinite(value) for value in advantage_values):
        raise ValueError("advantages must contain one finite value per sequence")
    return new, old, reference, advantage_values, _as_mask(response_mask, (len(new), len(new[0])))


def grpo_loss_reference(
    new_logprobs: Iterable[Iterable[float]],
    old_logprobs: Iterable[Iterable[float]],
    advantages: Iterable[float],
    response_mask: Iterable[Iterable[bool]],
    *,
    reference_logprobs: Iterable[Iterable[float]] | None = None,
    clip_range: float = 0.2,
    beta: float = 0.0,
) -> GrpoLossTerms:
    """Compute the clipped GRPO objective exactly, with no ML dependencies.

    The per-token policy term is `min(ratio * A, clip(ratio) * A)`.  `A` is
    sequence-level group-relative advantage and applies to all valid response
    tokens.  The non-negative KL estimator is `exp(ref - policy) - (ref -
    policy) - 1`, matching common LLM RL implementations.
    """
    new, old, reference, advantage_values, mask = _validate_inputs(
        new_logprobs, old_logprobs, reference_logprobs, advantages, response_mask, clip_range, beta
    )

    policy_sum = 0.0
    kl_sum = 0.0
    ratio_sum = 0.0
    clipped_count = 0
    token_count = 0
    for new_row, old_row, ref_row, advantage, mask_row in zip(
        new, old, reference or (None,) * len(new), advantage_values, mask, strict=True
    ):
        for new_logprob, old_logprob, ref_logprob, valid in zip(
            new_row, old_row, ref_row if ref_row is not None else (None,) * len(new_row), mask_row, strict=True
        ):
            if not valid:
                continue
            ratio = math.exp(new_logprob - old_logprob)
            clipped_ratio = min(max(ratio, 1.0 - clip_range), 1.0 + clip_range)
            policy_sum += min(ratio * advantage, clipped_ratio * advantage)
            ratio_sum += ratio
            clipped_count += int(ratio != clipped_ratio)
            if ref_logprob is not None:
                ref_minus_policy = ref_logprob - new_logprob
                kl_sum += math.exp(ref_minus_policy) - ref_minus_policy - 1.0
            token_count += 1

    if token_count == 0:
        raise ValueError("response_mask must include at least one valid token")
    policy_loss = -policy_sum / token_count
    kl_loss = kl_sum / token_count
    return GrpoLossTerms(
        total_loss=policy_loss + beta * kl_loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        mean_ratio=ratio_sum / token_count,
        clip_fraction=clipped_count / token_count,
        token_count=token_count,
    )


def torch_grpo_loss(
    new_logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    response_mask: Any,
    *,
    reference_logprobs: Any | None = None,
    clip_range: float = 0.2,
    beta: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    """Differentiable PyTorch GRPO loss.

    Args use shapes `[batch, response_tokens]`, except `advantages` with shape
    `[batch]`. The returned `loss` retains gradients; metrics are detached tensors
    so a trainer can reduce/log them without extending the autograd graph.
    """
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - exercised by user environments
        raise RuntimeError("torch_grpo_loss requires PyTorch; install the optional torch dependency") from error

    if clip_range < 0:
        raise ValueError("clip_range must be non-negative")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if new_logprobs.ndim != 2:
        raise ValueError("new_logprobs must have shape [batch, response_tokens]")
    if old_logprobs.shape != new_logprobs.shape or response_mask.shape != new_logprobs.shape:
        raise ValueError("old_logprobs and response_mask must match new_logprobs shape")
    if advantages.shape != (new_logprobs.shape[0],):
        raise ValueError("advantages must have shape [batch]")
    if reference_logprobs is not None and reference_logprobs.shape != new_logprobs.shape:
        raise ValueError("reference_logprobs must match new_logprobs shape")

    mask = response_mask.to(dtype=new_logprobs.dtype)
    token_count = mask.sum()
    if token_count.detach().item() <= 0:
        raise ValueError("response_mask must include at least one valid token")

    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    advantage = advantages[:, None]
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    policy_loss = -(surrogate * mask).sum() / token_count

    if reference_logprobs is None:
        kl_loss = torch.zeros((), device=new_logprobs.device, dtype=new_logprobs.dtype)
    else:
        ref_minus_policy = reference_logprobs - new_logprobs
        kl = torch.exp(ref_minus_policy) - ref_minus_policy - 1.0
        kl_loss = (kl * mask).sum() / token_count
    loss = policy_loss + beta * kl_loss

    with torch.no_grad():
        metrics = {
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "mean_ratio": (ratio * mask).sum() / token_count,
            "clip_fraction": (((ratio - clipped_ratio).abs() > 1e-7).to(mask.dtype) * mask).sum() / token_count,
            "token_count": token_count.detach(),
        }
    return loss, metrics
