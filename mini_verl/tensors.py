"""Tensorization at the causal-LM / trajectory boundary.

LLM RL losses are defined over *response* tokens, while a causal language model
returns logits for every position in `prompt + response`. This module centralizes
the off-by-one alignment so rollout and trainer backends share one contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .protocol import TrajectoryBatch, TrajectoryValidationError


@dataclass(frozen=True, slots=True)
class ResponseLogprobs:
    """Padded response-token log probabilities and the corresponding validity mask."""

    values: Any
    mask: Any


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - depends on environment
        raise RuntimeError("tensor helpers require PyTorch; install the optional torch dependency") from error
    return torch


def response_logprobs_from_logits(logits: Any, batch: TrajectoryBatch) -> ResponseLogprobs:
    """Extract `log p(response_token | preceding prompt/response tokens)`.

    `logits[b, t]` predicts the token at position `t + 1`.  For each trajectory,
    the first response token is consequently read from `prompt_length - 1`.
    Returned matrices are `[batch, max_response_length]`; entries made invalid by
    a trajectory's response mask or right padding are zero and do not contribute
    to an RL loss.
    """
    torch = _torch()
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if logits.shape[0] != len(batch.trajectories):
        raise ValueError("logits batch dimension must equal trajectory count")
    if logits.shape[1] < 1:
        raise ValueError("logits sequence dimension must be non-empty")

    max_response_length = max(len(trajectory.response_token_ids) for trajectory in batch.trajectories)
    values = torch.zeros(
        (len(batch.trajectories), max_response_length), device=logits.device, dtype=logits.dtype
    )
    mask = torch.zeros((len(batch.trajectories), max_response_length), device=logits.device, dtype=torch.bool)

    for row, trajectory in enumerate(batch.trajectories):
        prompt_length = len(trajectory.prompt_token_ids)
        response_length = len(trajectory.response_token_ids)
        final_token_position = prompt_length + response_length - 1
        if final_token_position >= logits.shape[1]:
            raise TrajectoryValidationError(
                "logits sequence length cannot score this trajectory's full prompt and response"
            )
        if max(trajectory.response_token_ids) >= logits.shape[2]:
            raise TrajectoryValidationError("response token id exceeds logits vocabulary size")

        prediction_logits = logits[row, prompt_length - 1 : final_token_position]
        response_tokens = torch.tensor(trajectory.response_token_ids, device=logits.device, dtype=torch.long)
        token_logprobs = torch.log_softmax(prediction_logits, dim=-1).gather(-1, response_tokens[:, None]).squeeze(-1)
        token_mask = torch.tensor(trajectory.response_mask, device=logits.device, dtype=torch.bool)
        values[row, :response_length] = token_logprobs * token_mask.to(logits.dtype)
        mask[row, :response_length] = token_mask

    return ResponseLogprobs(values=values, mask=mask)
