"""Hugging Face causal-LM rollout and GRPO training backends.

The initial implementation is deliberately synchronous and single-process.  It provides
the semantic contract needed by the controller today; a vLLM rollout backend or
distributed trainer can implement the same worker protocols later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from .algorithms.dpo import torch_dpo_loss
from .algorithms.grpo import torch_grpo_loss
from .batching import PackedTrajectoryBatch, sequence_length, length_bucket_batches
from .preference import PreferencePair, preference_pairs
from .protocol import Trajectory, TrajectoryBatch, TrajectoryValidationError
from .tensors import response_logprobs_from_logits


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Hugging Face backends require PyTorch") from error
    return torch


@dataclass(frozen=True, slots=True)
class PromptExample:
    """One prompt and metadata copied onto each of its sampled trajectories."""

    text: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("PromptExample.text must not be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class CausalLmInputs:
    """Right-padded complete sequences suitable for a causal LM forward pass."""

    input_ids: Any
    attention_mask: Any


@dataclass(frozen=True, slots=True)
class GenerationStageTimings:
    """Forward-time attribution inside one rollout's `model.generate` calls.

    Every call to ``generate`` has one initial model forward (prefill) followed
    by zero or more cache-backed forwards (decode).  These values intentionally
    exclude tokenization, sampling/bookkeeping, trajectory construction and the
    separate old-logprob recomputation forward.  They are optional diagnostic
    measurements, not a serving-engine latency API.
    """

    prefill_seconds: float
    decode_seconds: float
    prefill_forward_calls: int
    decode_forward_calls: int

    @property
    def forward_seconds(self) -> float:
        return self.prefill_seconds + self.decode_seconds


@dataclass(frozen=True, slots=True)
class RolloutStageTimings:
    """Optional forward-time attribution for one complete rollout.

    ``generation`` covers forwards issued by ``model.generate``.
    ``old_logprob_forward_seconds`` covers the separate full-sequence CausalLM
    forward that records sampled-policy logprobs in trajectories. Input packing
    and Python-side trajectory construction intentionally remain outside these
    fields.
    """

    generation: GenerationStageTimings
    old_logprob_forward_seconds: float
    prompt_batching: "PromptBatchingStats"


@dataclass(frozen=True, slots=True)
class PromptBatchingStats:
    """Prompt accounting plus a worst-case sequence-capacity observation.

    Counts include ``num_return_sequences`` because Transformers expands every
    prompt before the generation prefill.  The padding fields describe the
    prompt portion; ``max_batch_padded_sequence_tokens`` reserves configured
    worst-case generated length rather than sampled response length.
    """

    batch_count: int
    real_prompt_tokens: int
    padded_prompt_tokens: int
    max_batch_padded_prompt_tokens: int
    max_batch_padded_sequence_tokens: int

    @property
    def padding_ratio(self) -> float:
        if self.padded_prompt_tokens == 0:
            return 0.0
        return 1.0 - self.real_prompt_tokens / self.padded_prompt_tokens


class _GenerationForwardTimer:
    """Synchronously time first/subsequent forwards during one generate call."""

    def __init__(self, model: Any, *, device: Any) -> None:
        self._model = model
        self._device = device
        self._prefill_seconds = 0.0
        self._decode_seconds = 0.0
        self._prefill_calls = 0
        self._decode_calls = 0
        self._first_forward = True
        self._pending: list[tuple[bool, float]] = []
        self._pre_hook: Any | None = None
        self._post_hook: Any | None = None

    def _synchronize(self) -> None:
        torch = _torch()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _before_forward(self, module: Any, args: Any, kwargs: Any) -> None:
        del module, args, kwargs
        self._synchronize()
        self._pending.append((self._first_forward, perf_counter()))
        self._first_forward = False

    def _after_forward(self, module: Any, args: Any, kwargs: Any, output: Any) -> None:
        del module, args, kwargs, output
        if not self._pending:
            return
        is_prefill, started = self._pending.pop()
        self._synchronize()
        elapsed = perf_counter() - started
        if is_prefill:
            self._prefill_seconds += elapsed
            self._prefill_calls += 1
        else:
            self._decode_seconds += elapsed
            self._decode_calls += 1

    def start(self) -> None:
        if self._pre_hook is not None:
            raise RuntimeError("generation forward timer is already running")
        self._pre_hook = self._model.register_forward_pre_hook(self._before_forward, with_kwargs=True)
        self._post_hook = self._model.register_forward_hook(self._after_forward, with_kwargs=True)

    def stop(self) -> GenerationStageTimings:
        if self._pre_hook is None or self._post_hook is None:
            raise RuntimeError("generation forward timer was not started")
        self._pre_hook.remove()
        self._post_hook.remove()
        self._pre_hook = None
        self._post_hook = None
        return GenerationStageTimings(
            prefill_seconds=self._prefill_seconds,
            decode_seconds=self._decode_seconds,
            prefill_forward_calls=self._prefill_calls,
            decode_forward_calls=self._decode_calls,
        )


def causal_lm_inputs(batch: TrajectoryBatch, *, pad_token_id: int, device: Any) -> CausalLmInputs:
    """Pack each trajectory's `prompt + response` while preserving position zero."""
    torch = _torch()
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise ValueError("pad_token_id must be a non-negative integer")
    sequences = [trajectory.prompt_token_ids + trajectory.response_token_ids for trajectory in batch.trajectories]
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), max_length), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, : len(sequence)] = 1
    return CausalLmInputs(input_ids=input_ids, attention_mask=attention_mask)


def batch_logprobs(values: Sequence[Sequence[float]], *, device: Any, dtype: Any) -> Any:
    """Right-pad per-response logprobs to the trajectory batch's response width."""
    torch = _torch()
    width = max(len(row) for row in values)
    result = torch.zeros((len(values), width), device=device, dtype=dtype)
    for row_index, row in enumerate(values):
        result[row_index, : len(row)] = torch.tensor(row, device=device, dtype=dtype)
    return result


@dataclass(slots=True)
class HuggingFaceRolloutWorker:
    """Sample complete responses from a `transformers` causal-LM model.

    ``rollout_batch_size`` controls a correctness-first prompt micro-batch.
    Prompts are left-padded before decoder-only generation, so every generated
    row has the same prompt boundary while every returned trajectory retains its
    original unpadded prompt ids.  This is deliberately not continuous batching:
    a future vLLM/SGLang backend owns dynamic request scheduling.
    """

    model: Any
    tokenizer: Any
    prompts: Sequence[PromptExample]
    group_size: int
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True
    rollout_batch_size: int = 1
    bucket_prompts_by_length: bool = False
    rollout_max_padded_prompt_tokens: int | None = None
    rollout_max_padded_sequence_tokens: int | None = None
    collect_generation_timings: bool = False
    last_generation_timings: GenerationStageTimings | None = field(default=None, init=False)
    last_rollout_timings: RolloutStageTimings | None = field(default=None, init=False)
    last_prompt_batching_stats: PromptBatchingStats | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ValueError("prompts must not be empty")
        if self.group_size < 1 or self.max_new_tokens < 1 or self.rollout_batch_size < 1:
            raise ValueError("group_size, max_new_tokens, and rollout_batch_size must be positive")
        if self.rollout_max_padded_prompt_tokens is not None and self.rollout_max_padded_prompt_tokens < 1:
            raise ValueError("rollout_max_padded_prompt_tokens must be positive when set")
        if self.rollout_max_padded_sequence_tokens is not None and self.rollout_max_padded_sequence_tokens < 1:
            raise ValueError("rollout_max_padded_sequence_tokens must be positive when set")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature must be positive and top_p must be in (0, 1]")

    def rollout(self, *, policy_version: int) -> TrajectoryBatch:
        torch = _torch()
        device = next(self.model.parameters()).device
        pad_token_id = self._pad_token_id()
        trajectories: list[Trajectory] = []
        prefill_seconds = 0.0
        decode_seconds = 0.0
        prefill_calls = 0
        decode_calls = 0
        prompt_batch_count = 0
        real_prompt_tokens = 0
        padded_prompt_tokens = 0
        max_batch_padded_prompt_tokens = 0
        max_batch_padded_sequence_tokens = 0
        encoded_prompts = [
            (index, example, self._encode_prompt(example))
            for index, example in enumerate(self.prompts)
        ]
        self._validate_generation_context(encoded_prompts)
        if self.bucket_prompts_by_length:
            encoded_prompts.sort(key=lambda item: (-len(item[2]), item[0]))
        indexed_trajectories: list[tuple[int, int, Trajectory]] = []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for prompt_rows in self._prompt_batches(encoded_prompts):
                    prompt_ids, input_ids, attention_mask = self._prompt_batch(
                        prompt_rows, pad_token_id=pad_token_id, device=device
                    )
                    prompt_batch_count += 1
                    real_prompt_tokens += self.group_size * sum(len(row) for row in prompt_ids)
                    batch_padded_prompt_tokens = self.group_size * len(prompt_ids) * int(input_ids.shape[1])
                    padded_prompt_tokens += batch_padded_prompt_tokens
                    max_batch_padded_prompt_tokens = max(
                        max_batch_padded_prompt_tokens, batch_padded_prompt_tokens
                    )
                    max_batch_padded_sequence_tokens = max(
                        max_batch_padded_sequence_tokens,
                        self.group_size
                        * len(prompt_ids)
                        * (int(input_ids.shape[1]) + self.max_new_tokens),
                    )
                    timer = _GenerationForwardTimer(self.model, device=device) if self.collect_generation_timings else None
                    if timer is not None:
                        timer.start()
                    try:
                        generated = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            do_sample=self.do_sample,
                            temperature=self.temperature if self.do_sample else None,
                            top_p=self.top_p if self.do_sample else None,
                            num_return_sequences=self.group_size,
                            min_new_tokens=1,
                            max_new_tokens=self.max_new_tokens,
                            pad_token_id=pad_token_id,
                        )
                    finally:
                        if timer is not None:
                            timings = timer.stop()
                            prefill_seconds += timings.prefill_seconds
                            decode_seconds += timings.decode_seconds
                            prefill_calls += timings.prefill_forward_calls
                            decode_calls += timings.decode_forward_calls
                    prompt_width = input_ids.shape[1]
                    for generated_index, sequence in enumerate(generated):
                        local_prompt_index, sample_index = divmod(generated_index, self.group_size)
                        original_index, example, _ = prompt_rows[local_prompt_index]
                        response_ids = self._response_token_ids(
                            sequence, prompt_length=prompt_width, pad_token_id=pad_token_id
                        )
                        if not response_ids:
                            raise RuntimeError("generation returned an empty response despite min_new_tokens=1")
                        indexed_trajectories.append((
                            original_index,
                            sample_index,
                            Trajectory(
                                prompt_token_ids=prompt_ids[local_prompt_index],
                                response_token_ids=response_ids,
                                old_logprobs=tuple(0.0 for _ in response_ids),
                                policy_version=policy_version,
                                group_id=f"policy-{policy_version}-prompt-{original_index}",
                                prompt_text=example.text,
                                response_text=self.tokenizer.decode(response_ids, skip_special_tokens=True),
                                metadata={**example.metadata, "sample_index": sample_index},
                            ),
                        ))
        finally:
            self.model.train(was_training)
        trajectories = [trajectory for _, _, trajectory in sorted(indexed_trajectories)]
        prompt_batching = PromptBatchingStats(
            batch_count=prompt_batch_count,
            real_prompt_tokens=real_prompt_tokens,
            padded_prompt_tokens=padded_prompt_tokens,
            max_batch_padded_prompt_tokens=max_batch_padded_prompt_tokens,
            max_batch_padded_sequence_tokens=max_batch_padded_sequence_tokens,
        )
        self.last_prompt_batching_stats = prompt_batching
        generation_timings = (
            GenerationStageTimings(
                prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds,
                prefill_forward_calls=prefill_calls,
                decode_forward_calls=decode_calls,
            )
            if self.collect_generation_timings
            else None
        )
        self.last_generation_timings = generation_timings

        provisional = TrajectoryBatch.from_iterable(trajectories)
        inputs = causal_lm_inputs(provisional, pad_token_id=pad_token_id, device=device)
        old_logprob_forward_seconds = 0.0
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                if self.collect_generation_timings and device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = perf_counter()
                logits = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).logits
                if self.collect_generation_timings and device.type == "cuda":
                    torch.cuda.synchronize(device)
                old_logprob_forward_seconds = perf_counter() - started
                scores = response_logprobs_from_logits(logits, provisional).values
        finally:
            self.model.train(was_training)
        self.last_rollout_timings = (
            RolloutStageTimings(
                generation=generation_timings,
                old_logprob_forward_seconds=old_logprob_forward_seconds,
                prompt_batching=prompt_batching,
            )
            if generation_timings is not None
            else None
        )
        scored = []
        for row, trajectory in enumerate(provisional.trajectories):
            length = len(trajectory.response_token_ids)
            scored.append(
                Trajectory(
                    prompt_token_ids=trajectory.prompt_token_ids,
                    response_token_ids=trajectory.response_token_ids,
                    old_logprobs=tuple(float(value) for value in scores[row, :length].tolist()),
                    response_mask=trajectory.response_mask,
                    policy_version=trajectory.policy_version,
                    group_id=trajectory.group_id,
                    prompt_text=trajectory.prompt_text,
                    response_text=trajectory.response_text,
                    metadata=trajectory.metadata,
                )
            )
        return TrajectoryBatch.from_iterable(scored)

    def _prompt_batch(
        self,
        prompt_rows: Sequence[tuple[int, PromptExample, tuple[int, ...]]],
        *,
        pad_token_id: int,
        device: Any,
    ) -> tuple[tuple[tuple[int, ...], ...], Any, Any]:
        """Encode then left-pad a small decoder-only generation batch.

        Calling the tokenizer one prompt at a time keeps this worker compatible
        with minimal tokenizer facades used in tests while avoiding an implicit
        tokenizer-side padding policy.  Left padding and attention masks let
        ``generate`` align the final real prompt token across all rows.
        """
        torch = _torch()
        encoded_rows = [row for _, _, row in prompt_rows]
        width = max(len(row) for row in encoded_rows)
        input_ids = torch.full((len(encoded_rows), width), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(encoded_rows), width), dtype=torch.long, device=device)
        for index, row in enumerate(encoded_rows):
            input_ids[index, width - len(row) :] = torch.tensor(row, dtype=torch.long, device=device)
            attention_mask[index, width - len(row) :] = 1
        return tuple(encoded_rows), input_ids, attention_mask

    def _prompt_batches(
        self, encoded_prompts: Sequence[tuple[int, PromptExample, tuple[int, ...]]]
    ) -> Sequence[Sequence[tuple[int, PromptExample, tuple[int, ...]]]]:
        """Split scheduled prompts by count and optional expanded padding budget.

        The prompt budget is the prefill tensor token count after Transformers
        expands each prompt for ``num_return_sequences``:
        ``G * rows * max_prompt_length``.  The sequence budget additionally
        reserves the configured worst-case generated length:
        ``G * rows * (max_prompt_length + max_new_tokens)``.  Neither value is
        a byte-accurate memory prediction, but the latter is a useful static
        bound for decode/KV-cache growth under this fixed-length scheduler.
        """
        batches: list[list[tuple[int, PromptExample, tuple[int, ...]]]] = []
        current: list[tuple[int, PromptExample, tuple[int, ...]]] = []
        current_width = 0
        for row in encoded_prompts:
            if len(current) == self.rollout_batch_size:
                batches.append(current)
                current = []
                current_width = 0
            candidate_width = max(current_width, len(row[2]))
            candidate_prompt_tokens = self.group_size * (len(current) + 1) * candidate_width
            candidate_sequence_tokens = self.group_size * (len(current) + 1) * (candidate_width + self.max_new_tokens)
            exceeds_prompt_budget = (
                self.rollout_max_padded_prompt_tokens is not None
                and candidate_prompt_tokens > self.rollout_max_padded_prompt_tokens
            )
            exceeds_sequence_budget = (
                self.rollout_max_padded_sequence_tokens is not None
                and candidate_sequence_tokens > self.rollout_max_padded_sequence_tokens
            )
            if exceeds_prompt_budget or exceeds_sequence_budget:
                if not current:
                    raise ValueError(
                        "rollout padded-token budget is smaller than one expanded prompt/response; "
                        "increase the budget or reduce group_size, prompt length, or max_new_tokens"
                    )
                batches.append(current)
                current = []
                current_width = 0
                candidate_prompt_tokens = self.group_size * len(row[2])
                candidate_sequence_tokens = self.group_size * (len(row[2]) + self.max_new_tokens)
                exceeds_prompt_budget = (
                    self.rollout_max_padded_prompt_tokens is not None
                    and candidate_prompt_tokens > self.rollout_max_padded_prompt_tokens
                )
                exceeds_sequence_budget = (
                    self.rollout_max_padded_sequence_tokens is not None
                    and candidate_sequence_tokens > self.rollout_max_padded_sequence_tokens
                )
                if exceeds_prompt_budget or exceeds_sequence_budget:
                    raise ValueError(
                        "rollout padded-token budget is smaller than one expanded prompt/response; "
                        "increase the budget or reduce group_size, prompt length, or max_new_tokens"
                    )
            current.append(row)
            current_width = max(current_width, len(row[2]))
        if current:
            batches.append(current)
        return batches

    def _encode_prompt(self, example: PromptExample) -> tuple[int, ...]:
        encoded = self.tokenizer(example.text, return_tensors="pt", add_special_tokens=True)
        row = tuple(int(token) for token in encoded["input_ids"][0].tolist())
        if not row:
            raise RuntimeError("tokenizer returned an empty prompt")
        return row

    def _validate_generation_context(
        self, encoded_prompts: Sequence[tuple[int, PromptExample, tuple[int, ...]]]
    ) -> None:
        """Reject a statically impossible prompt before `generate` fails deep inside.

        Decoder-only Hugging Face configs do not use one uniform attribute name.
        When either common capacity field is present, preserve the worker's
        fixed-length contract by requiring room for all configured new tokens.
        Models that do not expose a static limit are left to their backend.
        """
        config = getattr(self.model, "config", None)
        capacity = None
        for name in ("max_position_embeddings", "n_positions"):
            value = getattr(config, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                capacity = value
                break
        if capacity is None:
            return
        for _, example, prompt_ids in encoded_prompts:
            required = len(prompt_ids) + self.max_new_tokens
            if required > capacity:
                raise ValueError(
                    f"prompt requires {required} positions (prompt={len(prompt_ids)}, "
                    f"max_new_tokens={self.max_new_tokens}) but model context capacity is {capacity}: "
                    f"{example.text!r}"
                )

    def _pad_token_id(self) -> int:
        token_id = self.tokenizer.pad_token_id
        if token_id is None:
            token_id = self.tokenizer.eos_token_id
        if token_id is None:
            raise ValueError("tokenizer needs a pad_token_id or eos_token_id")
        return int(token_id)

    def _response_token_ids(self, sequence: Any, *, prompt_length: int, pad_token_id: int) -> tuple[int, ...]:
        """Remove generate-time right padding without discarding a terminal EOS token."""
        response = [int(token) for token in sequence[prompt_length:].tolist()]
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None and int(eos_token_id) in response:
            return tuple(response[: response.index(int(eos_token_id)) + 1])
        while response and response[-1] == pad_token_id:
            response.pop()
        return tuple(response)


@dataclass(slots=True)
class HuggingFaceTrainerWorker:
    """Run one GRPO update over a causal-LM policy and optional frozen reference.

    Optional length-aware micro-batching accumulates gradients across packed
    trajectory groups, but still executes exactly one optimizer step for the
    caller's logical GRPO batch.  Each micro-loss is weighted by its effective
    response-token count so the result preserves the full-batch token-mean GRPO
    objective rather than averaging micro-batch means.
    """

    model: Any
    optimizer: Any
    pad_token_id: int
    reference_model: Any | None = None
    clip_range: float = 0.2
    beta: float = 0.0
    train_micro_batch_size: int | None = None
    train_max_padded_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.train_micro_batch_size is not None and self.train_micro_batch_size <= 0:
            raise ValueError("train_micro_batch_size must be positive when set")
        if self.train_max_padded_tokens is not None and self.train_max_padded_tokens <= 0:
            raise ValueError("train_max_padded_tokens must be positive when set")
        if self.reference_model is not None:
            self.reference_model.eval()
            for parameter in self.reference_model.parameters():
                parameter.requires_grad_(False)

    def train(self, batch: TrajectoryBatch, *, learner_policy_version: int) -> Mapping[str, float]:
        torch = _torch()
        if learner_policy_version < 0:
            raise ValueError("learner_policy_version must be non-negative")
        if len(batch.policy_versions) != 1:
            raise TrajectoryValidationError("trainer requires one rollout policy version per batch")
        if any(trajectory.advantage is None for trajectory in batch.trajectories):
            raise TrajectoryValidationError("trainer requires advantages from a reward worker")
        device = next(self.model.parameters()).device
        self.model.train()
        packed_batches = self._training_batches(batch)
        total_response_tokens = sum(
            sum(trajectory.response_mask)
            for packed in packed_batches
            for trajectory in packed.batch.trajectories
        )
        if total_response_tokens <= 0:
            raise ValueError("response_mask must include at least one valid token")
        if self.reference_model is not None:
            reference_device = next(self.reference_model.parameters()).device
            if reference_device != device:
                raise ValueError("policy and reference model must be on the same device in this backend")
        self.optimizer.zero_grad(set_to_none=True)
        metric_sums: dict[str, float] = {}
        for packed in packed_batches:
            micro_batch = packed.batch
            micro_response_tokens = sum(sum(trajectory.response_mask) for trajectory in micro_batch.trajectories)
            inputs = causal_lm_inputs(micro_batch, pad_token_id=self.pad_token_id, device=device)
            logits = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).logits
            policy_scores = response_logprobs_from_logits(logits, micro_batch)
            old_logprobs = batch_logprobs(
                [trajectory.old_logprobs for trajectory in micro_batch.trajectories],
                device=device,
                dtype=policy_scores.values.dtype,
            )
            advantages = torch.tensor(
                [float(trajectory.advantage) for trajectory in micro_batch.trajectories],
                dtype=policy_scores.values.dtype,
                device=device,
            )
            reference_logprobs = None
            if self.reference_model is not None:
                with torch.inference_mode():
                    reference_logits = self.reference_model(
                        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
                    ).logits
                    reference_logprobs = response_logprobs_from_logits(reference_logits, micro_batch).values
            loss, metrics = torch_grpo_loss(
                policy_scores.values,
                old_logprobs,
                advantages,
                policy_scores.mask,
                reference_logprobs=reference_logprobs,
                clip_range=self.clip_range,
                beta=self.beta,
            )
            (loss * (micro_response_tokens / total_response_tokens)).backward()
            for key, value in metrics.items():
                if key != "token_count":
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value.item()) * micro_response_tokens
        self.optimizer.step()
        real_sequence_tokens = sum(packed.real_sequence_tokens for packed in packed_batches)
        padded_sequence_tokens = sum(packed.padded_sequence_tokens for packed in packed_batches)
        aggregate = {key: value / total_response_tokens for key, value in metric_sums.items()}
        return {
            "loss": aggregate["policy_loss"] + self.beta * aggregate["kl_loss"],
            **aggregate,
            "token_count": float(total_response_tokens),
            "train_microbatch_count": float(len(packed_batches)),
            "train_real_sequence_tokens": float(real_sequence_tokens),
            "train_padded_sequence_tokens": float(padded_sequence_tokens),
            "train_padding_ratio": (padded_sequence_tokens - real_sequence_tokens) / padded_sequence_tokens,
        }

    def _training_batches(self, batch: TrajectoryBatch) -> tuple[PackedTrajectoryBatch, ...]:
        """Return one full batch by default, or deterministic length buckets."""
        trajectories = batch.trajectories
        max_length = max(sequence_length(trajectory) for trajectory in trajectories)
        if self.train_micro_batch_size is None and self.train_max_padded_tokens is None:
            return (
                PackedTrajectoryBatch(
                    batch=batch,
                    padded_sequence_tokens=max_length * len(trajectories),
                    real_sequence_tokens=sum(sequence_length(trajectory) for trajectory in trajectories),
                ),
            )
        max_batch_size = self.train_micro_batch_size or len(trajectories)
        max_padded_tokens = self.train_max_padded_tokens or max_batch_size * max_length
        return length_bucket_batches(
            trajectories, max_batch_size=max_batch_size, max_padded_tokens=max_padded_tokens
        )


@dataclass(slots=True)
class HuggingFaceDpoTrainerWorker:
    """Run one DPO update over preference pairs built from scored trajectories.

    Pairs are constructed online by `preference_pairs`: within each prompt
    group, the highest-reward response is chosen and the lowest-reward response
    is rejected.  Unlike the GRPO trainer, advantages and `old_logprobs` are
    deliberately unused — DPO compares the policy against the frozen reference
    model on both members of every pair, so `reference_model` is required.
    Each micro-loss is weighted by its pair count so the result preserves the
    full-batch pair-mean DPO objective, and exactly one optimizer step runs for
    the caller's logical batch.
    """

    model: Any
    optimizer: Any
    pad_token_id: int
    reference_model: Any
    beta: float = 0.1
    length_normalize: bool = False
    train_micro_batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.reference_model is None:
            raise ValueError("DPO requires a frozen reference model")
        if not self.beta > 0:
            raise ValueError("beta must be positive")
        if self.train_micro_batch_size is not None and self.train_micro_batch_size <= 0:
            raise ValueError("train_micro_batch_size must be positive when set")
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)

    def train(self, batch: TrajectoryBatch, *, learner_policy_version: int) -> Mapping[str, float]:
        torch = _torch()
        if learner_policy_version < 0:
            raise ValueError("learner_policy_version must be non-negative")
        if len(batch.policy_versions) != 1:
            raise TrajectoryValidationError("trainer requires one rollout policy version per batch")
        pairs = preference_pairs(batch)
        if not pairs:
            raise ValueError("batch must contain at least one valid preference pair")
        device = next(self.model.parameters()).device
        reference_device = next(self.reference_model.parameters()).device
        if reference_device != device:
            raise ValueError("policy and reference model must be on the same device in this backend")
        self.model.train()
        pair_batches = self._pair_batches(pairs)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss_sum = 0.0
        metric_sums: dict[str, float] = {}
        for pair_batch in pair_batches:
            micro_pair_count = len(pair_batch)
            micro_batch = TrajectoryBatch(
                tuple(pair.chosen for pair in pair_batch) + tuple(pair.rejected for pair in pair_batch)
            )
            inputs = causal_lm_inputs(micro_batch, pad_token_id=self.pad_token_id, device=device)
            logits = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).logits
            policy_scores = response_logprobs_from_logits(logits, micro_batch)
            with torch.inference_mode():
                reference_logits = self.reference_model(
                    input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
                ).logits
                reference_scores = response_logprobs_from_logits(reference_logits, micro_batch)
            loss, metrics = torch_dpo_loss(
                policy_scores.values,
                reference_scores.values,
                policy_scores.mask,
                beta=self.beta,
                length_normalize=self.length_normalize,
            )
            (loss * (micro_pair_count / len(pairs))).backward()
            total_loss_sum += float(loss.detach().item()) * micro_pair_count
            for key, value in metrics.items():
                if key != "pair_count":
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value.item()) * micro_pair_count
        self.optimizer.step()
        aggregate = {key: value / len(pairs) for key, value in metric_sums.items()}
        return {
            "loss": total_loss_sum / len(pairs),
            **aggregate,
            "pair_count": float(len(pairs)),
            "train_microbatch_count": float(len(pair_batches)),
        }

    def _pair_batches(self, pairs: tuple[PreferencePair, ...]) -> tuple[tuple[PreferencePair, ...], ...]:
        """Split pairs into deterministic micro-batches of at most `train_micro_batch_size` pairs."""
        if self.train_micro_batch_size is None:
            return (pairs,)
        return tuple(
            pairs[index : index + self.train_micro_batch_size]
            for index in range(0, len(pairs), self.train_micro_batch_size)
        )
