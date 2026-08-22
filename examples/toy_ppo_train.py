"""A minimal PPO actor--critic run on the same categorical task as toy GRPO.

Each of 16 prompts has one correct answer token.  PPO samples eight answers per
prompt, uses a table of 16 Critic values to calculate GAE, then optimizes the
same clipped policy objective as GRPO plus a value-regression loss.  Episodes
end after one answer, so GAE is visibly ``reward - V(prompt)`` here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_verl.algorithms.ppo import generalized_advantage_estimate, torch_ppo_loss
from mini_verl.config import RunConfig, seed_everything


@dataclass(frozen=True, slots=True)
class ToyPpoRunResult:
    initial_pass_at_1: float
    final_pass_at_1: float
    final_reward: float
    final_actor_loss: float
    final_value_loss: float


def _pass_at_1(logits: torch.Tensor, correct_tokens: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == correct_tokens).float().mean().item())


def _rollout(
    logits: torch.Tensor, values: torch.Tensor, correct_tokens: torch.Tensor, *, samples_per_prompt: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits.detach(), dim=-1)
    actions = torch.multinomial(probabilities, num_samples=samples_per_prompt, replacement=True).reshape(-1)
    prompt_ids = torch.arange(logits.shape[0], device=logits.device).repeat_interleave(samples_per_prompt)
    old_logprobs = torch.log(probabilities[prompt_ids, actions])
    rewards = (actions == correct_tokens[prompt_ids]).to(dtype=logits.dtype)
    old_values = values.detach()[prompt_ids]
    return prompt_ids, actions, old_logprobs, rewards, old_values


def run(*, device: str | torch.device | None = None, iterations: int = 100, seed: int = 7) -> ToyPpoRunResult:
    """Train the tiny actor and Critic; returns the shared pass@1 metric."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(RunConfig(seed=seed, device=str(selected_device)))
    prompt_count, vocab_size, samples_per_prompt = 16, 8, 8
    correct_tokens = torch.arange(prompt_count, device=selected_device) % vocab_size
    logits = torch.nn.Parameter(torch.zeros(prompt_count, vocab_size, device=selected_device))
    critic_values = torch.nn.Parameter(torch.zeros(prompt_count, device=selected_device))
    optimizer = torch.optim.AdamW([logits, critic_values], lr=0.18, weight_decay=0.0)
    initial = _pass_at_1(logits.detach(), correct_tokens)
    final_reward = final_actor_loss = final_value_loss = float("nan")

    for _ in range(iterations):
        prompt_ids, actions, old_logprobs, rewards, old_values = _rollout(
            logits, critic_values, correct_tokens, samples_per_prompt=samples_per_prompt
        )
        # Every categorical answer is terminal.  Keeping this generic GAE call
        # makes the PPO-specific value baseline explicit rather than hidden.
        advantages, returns = generalized_advantage_estimate(
            rewards.detach().cpu().tolist(),
            old_values.detach().cpu().tolist(),
            [True] * rewards.numel(),
            gamma=1.0,
            gae_lambda=1.0,
        )
        advantage_tensor = torch.tensor(advantages, device=selected_device, dtype=logits.dtype)
        return_tensor = torch.tensor(returns, device=selected_device, dtype=logits.dtype)
        final_reward = float(rewards.mean().item())

        # PPO conventionally performs several clipped updates on the same rollout.
        for _ in range(2):
            new_logprobs = torch.log_softmax(logits[prompt_ids], dim=-1).gather(1, actions[:, None]).squeeze(1)
            loss, metrics = torch_ppo_loss(
                new_logprobs, old_logprobs, advantage_tensor, critic_values[prompt_ids], return_tensor, clip_range=0.2
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        final_actor_loss = float(metrics["actor_loss"].item())
        final_value_loss = float(metrics["value_loss"].item())

    return ToyPpoRunResult(
        initial, _pass_at_1(logits.detach(), correct_tokens), final_reward, final_actor_loss, final_value_loss
    )


def main() -> None:
    result = run()
    print(f"initial_pass@1={result.initial_pass_at_1:.3f}")
    print(f"final_pass@1={result.final_pass_at_1:.3f}")
    print(f"last_rollout_mean_reward={result.final_reward:.3f}")
    print(f"last_ppo_actor_loss={result.final_actor_loss:.6f}")
    print(f"last_ppo_value_loss={result.final_value_loss:.6f}")
    if result.final_pass_at_1 <= result.initial_pass_at_1:
        raise RuntimeError("toy PPO run did not improve policy pass@1")


if __name__ == "__main__":
    main()
