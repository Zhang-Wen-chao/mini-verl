"""DPO's sequence-level preference objective.

This module has two implementations of the same loss. `dpo_loss_reference` is
dependency-free and deliberately scalar, which makes its numerical behavior easy
to test. `torch_dpo_loss` is the differentiable implementation used by a trainer
once PyTorch is installed.

Rows follow the concatenated layout used by DPO trainers: row `i` for
`i < pair_count` is the chosen response of pair `i`, and row `i + pair_count`
is its rejected response. This mirrors a single forward pass over a
chosen/rejected concatenated batch, so a trainer computes the policy and
reference logprobs for both members of every pair at once.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class DpoLossTerms:
    """Scalar diagnostics from one DPO loss calculation.

    `chosen_reward` and `rejected_reward` are the implicit DPO rewards
    `beta * (policy_logprob - reference_logprob)`, averaged over pairs.
    """

    total_loss: float
    chosen_reward: float
    rejected_reward: float
    reward_margin: float
    accuracy: float
    pair_count: int


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
    reference_logprobs: Iterable[Iterable[float]],
    response_mask: Iterable[Iterable[bool]],
    beta: float,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...], tuple[tuple[bool, ...], ...], int]:
    if not beta > 0:
        raise ValueError("beta must be positive")
    new = _as_rows("new_logprobs", new_logprobs)
    if len(new) % 2 != 0:
        raise ValueError("new_logprobs must contain an even number of rows: one chosen and one rejected per pair")
    reference = _as_rows("reference_logprobs", reference_logprobs)
    if (len(reference), len(reference[0])) != (len(new), len(new[0])):
        raise ValueError("reference_logprobs must match new_logprobs shape")
    mask = _as_mask(response_mask, (len(new), len(new[0])))
    if any(not any(mask_row) for mask_row in mask):
        raise ValueError("response_mask must include at least one valid token in every row")
    return new, reference, mask, len(new) // 2


def _sequence_logprob(row: tuple[float, ...], mask_row: tuple[bool, ...], *, length_normalize: bool) -> float:
    total = 0.0
    token_count = 0
    for logprob, valid in zip(row, mask_row, strict=True):
        if not valid:
            continue
        total += logprob
        token_count += 1
    return total / token_count if length_normalize else total


def _softplus(value: float) -> float:
    if value > 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def dpo_loss_reference(
    new_logprobs: Iterable[Iterable[float]],
    reference_logprobs: Iterable[Iterable[float]],
    response_mask: Iterable[Iterable[bool]],
    *,
    beta: float = 0.1,
    length_normalize: bool = False,
) -> DpoLossTerms:
    """Compute the sequence-level DPO objective exactly, with no ML dependencies.

    The implicit reward of a response is `beta * (policy_logprob -
    reference_logprob)` over its valid tokens, and the loss is
    `-log(sigmoid(chosen_reward - rejected_reward))`.  `length_normalize=True`
    divides every sequence logprob sum by its valid token count.
    """
    new, reference, mask, pair_count = _validate_inputs(new_logprobs, reference_logprobs, response_mask, beta)

    total_loss = 0.0
    chosen_reward_sum = 0.0
    rejected_reward_sum = 0.0
    preferred_count = 0
    for pair_index in range(pair_count):
        chosen_reward = beta * (
            _sequence_logprob(new[pair_index], mask[pair_index], length_normalize=length_normalize)
            - _sequence_logprob(reference[pair_index], mask[pair_index], length_normalize=length_normalize)
        )
        rejected_reward = beta * (
            _sequence_logprob(new[pair_count + pair_index], mask[pair_count + pair_index], length_normalize=length_normalize)
            - _sequence_logprob(reference[pair_count + pair_index], mask[pair_count + pair_index], length_normalize=length_normalize)
        )
        margin = chosen_reward - rejected_reward
        total_loss += _softplus(-margin)
        chosen_reward_sum += chosen_reward
        rejected_reward_sum += rejected_reward
        preferred_count += int(margin > 0.0)

    return DpoLossTerms(
        total_loss=total_loss / pair_count,
        chosen_reward=chosen_reward_sum / pair_count,
        rejected_reward=rejected_reward_sum / pair_count,
        reward_margin=(chosen_reward_sum - rejected_reward_sum) / pair_count,
        accuracy=preferred_count / pair_count,
        pair_count=pair_count,
    )


def torch_dpo_loss(
    new_logprobs: Any,
    reference_logprobs: Any,
    response_mask: Any,
    *,
    beta: float = 0.1,
    length_normalize: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Differentiable PyTorch DPO loss.

    Args use the concatenated `[2 * pairs, response_tokens]` layout described in
    the module docstring. The returned `loss` retains gradients; metrics are
    detached tensors so a trainer can reduce/log them without extending the
    autograd graph.
    """
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - exercised by user environments
        raise RuntimeError("torch_dpo_loss requires PyTorch; install the optional torch dependency") from error

    if not beta > 0:
        raise ValueError("beta must be positive")
    if new_logprobs.ndim != 2:
        raise ValueError("new_logprobs must have shape [pairs * 2, response_tokens]")
    row_count = new_logprobs.shape[0]
    if row_count % 2 != 0 or row_count == 0:
        raise ValueError("new_logprobs must contain an even number of rows: one chosen and one rejected per pair")
    if reference_logprobs.shape != new_logprobs.shape or response_mask.shape != new_logprobs.shape:
        raise ValueError("reference_logprobs and response_mask must match new_logprobs shape")

    mask = response_mask.to(dtype=new_logprobs.dtype)
    valid_counts = mask.sum(dim=1)
    if int(valid_counts.min().detach().item()) <= 0:
        raise ValueError("response_mask must include at least one valid token in every row")

    policy_sums = (new_logprobs * mask).sum(dim=1)
    reference_sums = (reference_logprobs * mask).sum(dim=1)
    if length_normalize:
        policy_sums = policy_sums / valid_counts
        reference_sums = reference_sums / valid_counts

    pair_count = row_count // 2
    chosen_reward = beta * (policy_sums[:pair_count] - reference_sums[:pair_count])
    rejected_reward = beta * (policy_sums[pair_count:] - reference_sums[pair_count:])
    margin = chosen_reward - rejected_reward
    loss = torch.nn.functional.softplus(-margin).mean()

    with torch.no_grad():
        metrics = {
            "chosen_reward": chosen_reward.detach().mean(),
            "rejected_reward": rejected_reward.detach().mean(),
            "reward_margin": margin.detach().mean(),
            "accuracy": (margin > 0).to(mask.dtype).mean(),
            "pair_count": torch.tensor(pair_count, device=new_logprobs.device),
        }
    return loss, metrics
