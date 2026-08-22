"""A small, testable PPO actor--critic objective.

This module intentionally models one scalar action/log-probability per sampled
trajectory.  That is enough to expose PPO's two distinct pieces: the clipped
policy update and the Critic/value regression.  LLM token masking and reference
KL follow the same semantics as :mod:`mini_verl.algorithms.grpo`; production
multi-token PPO needs additional batching and sequence plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class PpoLossTerms:
    """Scalar diagnostics from one PPO actor--critic loss calculation."""

    total_loss: float
    actor_loss: float
    value_loss: float
    kl_loss: float
    mean_ratio: float
    clip_fraction: float
    sample_count: int


def _vector(name: str, values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def generalized_advantage_estimate(
    rewards: Iterable[float],
    values: Iterable[float],
    dones: Iterable[bool],
    *,
    bootstrap_value: float = 0.0,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return GAE advantages and value targets for one trajectory.

    ``delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`` and GAE recursively
    accumulates those residuals until a terminal state.  ``bootstrap_value`` is
    only used after a non-terminal final step.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if not math.isfinite(bootstrap_value):
        raise ValueError("bootstrap_value must be finite")
    reward_values = _vector("rewards", rewards)
    value_values = _vector("values", values)
    done_values = tuple(dones)
    if len(value_values) != len(reward_values) or len(done_values) != len(reward_values):
        raise ValueError("rewards, values, and dones must have the same length")
    if any(not isinstance(done, bool) for done in done_values):
        raise ValueError("dones must contain only booleans")

    advantages = [0.0] * len(reward_values)
    next_advantage = 0.0
    for index in range(len(reward_values) - 1, -1, -1):
        nonterminal = 0.0 if done_values[index] else 1.0
        next_value = bootstrap_value if index == len(reward_values) - 1 else value_values[index + 1]
        delta = reward_values[index] + gamma * nonterminal * next_value - value_values[index]
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[index] = next_advantage
    returns = tuple(advantage + value for advantage, value in zip(advantages, value_values, strict=True))
    return tuple(advantages), returns


def ppo_loss_reference(
    new_logprobs: Iterable[float],
    old_logprobs: Iterable[float],
    advantages: Iterable[float],
    new_values: Iterable[float],
    returns: Iterable[float],
    *,
    reference_logprobs: Iterable[float] | None = None,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    beta: float = 0.0,
) -> PpoLossTerms:
    """Compute PPO's clipped actor loss plus a squared-error Critic loss.

    The reported ``value_loss`` already includes the conventional ``0.5``
    multiplier; ``value_coef`` controls how much it contributes to total loss.
    """
    if clip_range < 0.0:
        raise ValueError("clip_range must be non-negative")
    if value_coef < 0.0 or beta < 0.0:
        raise ValueError("value_coef and beta must be non-negative")
    new = _vector("new_logprobs", new_logprobs)
    old = _vector("old_logprobs", old_logprobs)
    advantage_values = _vector("advantages", advantages)
    predicted_values = _vector("new_values", new_values)
    target_values = _vector("returns", returns)
    count = len(new)
    if any(len(values) != count for values in (old, advantage_values, predicted_values, target_values)):
        raise ValueError("all PPO inputs must have the same length")
    reference = None if reference_logprobs is None else _vector("reference_logprobs", reference_logprobs)
    if reference is not None and len(reference) != count:
        raise ValueError("reference_logprobs must match new_logprobs length")

    actor_sum = value_sum = kl_sum = ratio_sum = 0.0
    clipped_count = 0
    for index, (new_logprob, old_logprob, advantage, prediction, target) in enumerate(
        zip(new, old, advantage_values, predicted_values, target_values, strict=True)
    ):
        ratio = math.exp(new_logprob - old_logprob)
        clipped_ratio = min(max(ratio, 1.0 - clip_range), 1.0 + clip_range)
        actor_sum += min(ratio * advantage, clipped_ratio * advantage)
        value_sum += 0.5 * (prediction - target) ** 2
        ratio_sum += ratio
        clipped_count += int(ratio != clipped_ratio)
        if reference is not None:
            ref_minus_policy = reference[index] - new_logprob
            kl_sum += math.exp(ref_minus_policy) - ref_minus_policy - 1.0

    actor_loss = -actor_sum / count
    value_loss = value_sum / count
    kl_loss = kl_sum / count
    return PpoLossTerms(
        total_loss=actor_loss + value_coef * value_loss + beta * kl_loss,
        actor_loss=actor_loss,
        value_loss=value_loss,
        kl_loss=kl_loss,
        mean_ratio=ratio_sum / count,
        clip_fraction=clipped_count / count,
        sample_count=count,
    )


def torch_ppo_loss(
    new_logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    new_values: Any,
    returns: Any,
    *,
    reference_logprobs: Any | None = None,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    beta: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    """Differentiable one-action-per-trajectory PPO actor--critic loss."""
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("torch_ppo_loss requires PyTorch; install the optional torch dependency") from error

    if clip_range < 0.0:
        raise ValueError("clip_range must be non-negative")
    if value_coef < 0.0 or beta < 0.0:
        raise ValueError("value_coef and beta must be non-negative")
    if new_logprobs.ndim != 1:
        raise ValueError("new_logprobs must have shape [batch]")
    expected_shape = new_logprobs.shape
    if any(value.shape != expected_shape for value in (old_logprobs, advantages, new_values, returns)):
        raise ValueError("all PPO tensors must have shape [batch]")
    if new_logprobs.numel() == 0:
        raise ValueError("PPO batch must be non-empty")
    if reference_logprobs is not None and reference_logprobs.shape != expected_shape:
        raise ValueError("reference_logprobs must have shape [batch]")

    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    actor_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
    value_loss = 0.5 * (new_values - returns).square().mean()
    if reference_logprobs is None:
        kl_loss = torch.zeros((), device=new_logprobs.device, dtype=new_logprobs.dtype)
    else:
        ref_minus_policy = reference_logprobs - new_logprobs
        kl_loss = (torch.exp(ref_minus_policy) - ref_minus_policy - 1.0).mean()
    loss = actor_loss + value_coef * value_loss + beta * kl_loss

    with torch.no_grad():
        metrics = {
            "actor_loss": actor_loss.detach(),
            "value_loss": value_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "mean_ratio": ratio.mean(),
            "clip_fraction": ((ratio - clipped_ratio).abs() > 1e-7).to(new_logprobs.dtype).mean(),
            "sample_count": torch.tensor(new_logprobs.numel(), device=new_logprobs.device),
        }
    return loss, metrics

