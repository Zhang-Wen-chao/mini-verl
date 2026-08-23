"""Rollout worker backed by the mini-vllm engine (self-written vLLM subset).

Implements mini-verl's RolloutWorker protocol using mini_vllm.Engine, so the
RL loop (mini-verl) can generate rollouts through our own paged-attention
engine instead of Hugging Face / vLLM. This is step ② of the "all-self-written
stack" plan: mini-verl -> mini-vllm (rollout) + mini-megatron (train).

old_logprobs are recomputed after generation with a forward pass over the full
sequence (same as HuggingFaceRolloutWorker) so the GRPO ratio is meaningful —
placeholder zeros would make ratio = exp(new_logprobs) and clip everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from mini_verl.protocol import Trajectory, TrajectoryBatch, TrajectoryValidationError
from mini_verl.workers import RolloutWorker


@dataclass(slots=True)
class MiniVllmRolloutWorker(RolloutWorker):
    """Generate group rollouts through the mini-vllm engine.

    ``engine`` is a mini_vllm.engine.Engine instance (already constructed with
    the model and sampling config). Each prompt in ``prompts`` is rolled out
    ``group_size`` times with stochastic sampling (temperature>0) so GRPO gets
    a diverse group. ``max_new_tokens`` caps generation length.
    """

    engine: Any
    tokenizer: Any  # token<->id mapping compatible with the engine's model
    prompts: Sequence[Any]  # mini_verl.hf.PromptExample-like: .text + .metadata
    group_size: int = 4
    max_new_tokens: int = 32

    def _encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text)
        return ids

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def _old_logprobs(self, prompt_ids: list[int], response_ids: list[int]) -> list[float]:
        """Recompute per-token logprobs for the response via a full forward.

        Uses the adapter's model (engine.model.model is the raw GPT /
        transformers model). Returns log p(response_t | prefix) per response
        token, matching response_logprobs_from_logits semantics.
        """
        model = self.engine.model.model
        full = torch.tensor(prompt_ids + response_ids, device=self.device).unsqueeze(0)
        try:
            out = model(input_ids=full, attention_mask=torch.ones_like(full))
        except TypeError:
            out = model(input_ids=full)
        logits = getattr(out, "logits", out)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits[0]  # [T, V]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        # log p(response_t) = logits at position (prompt_len-1 + t), token response_t
        prompt_len = len(prompt_ids)
        probs = []
        for t, tok in enumerate(response_ids):
            pos = prompt_len - 1 + t
            if pos < logits.shape[0]:
                probs.append(float(logprobs[pos, tok].item()))
            else:
                probs.append(0.0)
        return probs

    def rollout(self, *, policy_version: int) -> TrajectoryBatch:
        if self.engine.temperature <= 0:
            raise TrajectoryValidationError(
                "mini-vllm rollout requires stochastic sampling (temperature > 0) "
                "so GRPO groups are diverse"
            )
        self.device = next(self.engine.model.model.parameters()).device
        trajectories: list[Trajectory] = []
        for prompt in self.prompts:
            prompt_ids = self._encode(prompt.text)
            for i in range(self.group_size):
                req = self.engine.add_request(
                    torch.tensor(prompt_ids), max_new_tokens=self.max_new_tokens
                )
                while self.engine.has_requests():
                    self.engine.step()
                full = self.engine.output(req)
                response_ids = full[len(prompt_ids):]
                response_text = self._decode(response_ids)
                old_logprobs = self._old_logprobs(prompt_ids, response_ids)
                trajectories.append(
                    Trajectory(
                        prompt_token_ids=tuple(prompt_ids),
                        response_token_ids=tuple(response_ids),
                        old_logprobs=tuple(old_logprobs),
                        policy_version=policy_version,
                        group_id=prompt.metadata.get("group_id", f"p{id(prompt)}"),
                        response_text=response_text,
                        prompt_text=prompt.text,
                        metadata={**prompt.metadata, "sample_index": i},
                    )
                )
        return TrajectoryBatch.from_iterable(trajectories)
