"""End-to-end wiring test: mini-verl Controller + mini-vllm Engine rollout.

Validates that the RL loop can drive our own paged-attention engine as the
rollout backend (step ② of the all-self-written stack). Uses mini-vllm's
TinyTransformer toy model: prompts are raw id sequences, reward = 1 when the
generated response ends with a target token id. GRPO should push the policy
toward emitting the target token.

Run with the PyTorch env that has both mini_verl and mini_vllm importable:
  python examples/toy_minivllm_grpo.py
"""

from __future__ import annotations

import sys

import torch

from mini_verl.controller import Controller
from mini_verl.hf import HuggingFaceTrainerWorker, PromptExample
from mini_verl.protocol import Trajectory
from mini_verl.workers import RuleRewardWorker
from mini_verl.rollout_minivllm import MiniVllmRolloutWorker


class IdTokenizer:
    """Trivial tokenizer: ids are themselves; decode maps id->str(id)."""

    def encode(self, text):
        return [int(t) for t in text.split(",") if t.strip()]

    def decode(self, ids):
        return ",".join(str(i) for i in ids)


class LogitsWrapper:
    """Wrap a raw logits tensor so trainer sees `.logits` (transformers style)."""

    def __init__(self, logits):
        self.logits = logits


class TinyTransformersAdapter:
    """Adapt mini-vllm's TinyTransformer to the trainer's transformers-style
    call: forward(input_ids, attention_mask) -> object with .logits.

    Reconstructs full sequences (prompt + response) and returns the model's
    logits so response_logprobs_from_logits can compute per-token logprobs.
    """

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def parameters(self):
        return self.model.parameters()

    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()

    @property
    def training(self):
        return self.model.training

    def forward(self, input_ids, attention_mask=None):
        # input_ids: [B, T]; process each row densely and return [B, T, V]
        logits = []
        for row in input_ids:
            toks = row[row != 0]  # drop padding (left-padded)
            logits.append(self.model.dense_forward(toks))
        max_t = max(l.shape[0] for l in logits)
        out = torch.zeros(len(input_ids), max_t, self.model.vocab_size,
                          device=self.device)
        for i, l in enumerate(logits):
            out[i, -l.shape[0]:] = l
        return LogitsWrapper(out)


def main() -> None:
    from mini_vllm.engine import Engine
    from mini_vllm.model_runner import TinyTransformer

    torch.manual_seed(42)
    model = TinyTransformer(vocab_size=64, d_model=32, n_layers=2, n_heads=4)
    engine = Engine(model, temperature=1.0, top_p=1.0, block_size=8, num_blocks=32)
    tokenizer = IdTokenizer()

    # target token = 7; prompts encourage ending with it
    prompts = [
        PromptExample("5,3,7", {"group_id": "p0", "target": 7}),
        PromptExample("2,4,7", {"group_id": "p1", "target": 7}),
        PromptExample("1,6,7", {"group_id": "p2", "target": 7}),
    ]

    def reward(t: Trajectory) -> float:
        return float(t.response_token_ids and t.response_token_ids[-1] == 7)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    adapter = TinyTransformersAdapter(model)
    controller = Controller(
        rollout_worker=MiniVllmRolloutWorker(
            engine=engine, tokenizer=tokenizer, prompts=prompts,
            group_size=4, max_new_tokens=16,
        ),
        reward_worker=RuleRewardWorker(reward),
        trainer_worker=HuggingFaceTrainerWorker(
            model=adapter, optimizer=optimizer, pad_token_id=0,
        ),
    )

    rewards = []
    for it in range(30):
        result = controller.run_iteration()
        rewards.append(result.mean_reward)
        if it % 5 == 0 or it == 29:
            print(
                f"iter {it:3d} | reward={result.mean_reward:.3f} | "
                f"loss={result.metrics['loss']:.5f} | "
                f"clip={result.metrics['clip_fraction']:.3f}",
                flush=True,
            )

    print(f"\nreward start={rewards[0]:.3f} end={rewards[-1]:.3f}")
    if rewards[-1] > rewards[0]:
        print("PASS: GRPO pushed the policy toward the target token via mini-vllm rollout")
    else:
        print("FAIL: reward did not improve")
        sys.exit(1)


if __name__ == "__main__":
    main()
