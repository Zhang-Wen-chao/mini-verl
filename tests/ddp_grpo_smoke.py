"""Two-rank CUDA smoke for DDP gradient synchronization through the GRPO loss.

Run with torchrun. Each rank receives different actions and advantages; after an
optimizer step, DDP must leave every policy replica numerically identical.
"""

from __future__ import annotations

import json

import torch
from torch.nn.parallel import DistributedDataParallel

from mini_verl.algorithms.grpo import torch_grpo_loss
from mini_verl.distributed import destroy_distributed, initialize_distributed, mean_across_ranks


class TokenPolicy(torch.nn.Module):
    def __init__(self, prompt_count: int, vocab_size: int) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(prompt_count, vocab_size))

    def forward(self, prompt_ids: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.logits[prompt_ids], dim=-1).gather(1, action_ids[:, None])


def main() -> None:
    context = initialize_distributed()
    try:
        torch.manual_seed(19)
        policy = DistributedDataParallel(
            TokenPolicy(prompt_count=4, vocab_size=5).to(context.device),
            device_ids=[context.local_rank],
        )
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.3)

        prompt_ids = torch.tensor([context.rank, context.rank + 1], device=context.device)
        action_ids = torch.tensor([context.rank + 1, context.rank + 2], device=context.device)
        advantages = torch.tensor([1.0, -0.5], device=context.device)
        old_logprobs = policy(prompt_ids, action_ids).detach()
        new_logprobs = policy(prompt_ids, action_ids)
        loss, metrics = torch_grpo_loss(
            new_logprobs,
            old_logprobs,
            advantages,
            torch.ones_like(new_logprobs, dtype=torch.bool),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        parameter = policy.module.logits.detach()
        gathered = [torch.empty_like(parameter) for _ in range(context.world_size)]
        torch.distributed.all_gather(gathered, parameter)
        if not all(torch.allclose(parameter, other, atol=1e-7, rtol=0.0) for other in gathered):
            raise AssertionError("DDP replicas diverged after GRPO update")
        mean_loss = mean_across_ranks(loss)
        if context.is_primary:
            print(json.dumps({
                "world_size": context.world_size,
                "mean_loss": round(float(mean_loss.item()), 8),
                "token_count_per_rank": int(metrics["token_count"].item()),
                "replicas_synchronized": True,
            }, sort_keys=True))
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
