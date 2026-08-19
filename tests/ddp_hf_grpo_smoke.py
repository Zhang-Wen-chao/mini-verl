"""Two-rank DDP smoke using the real HuggingFace CausalLM trainer backend.

Run with torchrun on CUDA. Each rank trains a tiny, identically initialized GPT-2
on a different GRPO trajectory batch. DDP must synchronize the causal-LM policy
parameters after the update.
"""

from __future__ import annotations

import json

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import GPT2Config, GPT2LMHeadModel

from mini_verl.distributed import destroy_distributed, initialize_distributed, mean_across_ranks
from mini_verl.hf import HuggingFaceTrainerWorker
from mini_verl.protocol import Trajectory, TrajectoryBatch


def local_batch(rank: int) -> TrajectoryBatch:
    return TrajectoryBatch((
        Trajectory(
            prompt_token_ids=(1, 2),
            response_token_ids=(3 + rank, 5),
            old_logprobs=(-2.0, -2.0),
            policy_version=0,
            group_id=f"rank-{rank}",
            advantage=1.0,
        ),
        Trajectory(
            prompt_token_ids=(1, 2),
            response_token_ids=(6 + rank,),
            old_logprobs=(-2.0,),
            policy_version=0,
            group_id=f"rank-{rank}",
            advantage=-1.0,
        ),
    ))


def main() -> None:
    context = initialize_distributed()
    try:
        torch.manual_seed(23)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=16,
                n_positions=16,
                n_embd=16,
                n_layer=1,
                n_head=1,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=1,
            )
        ).to(context.device)
        model = DistributedDataParallel(model, device_ids=[context.local_rank])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        trainer = HuggingFaceTrainerWorker(model=model, optimizer=optimizer, pad_token_id=0)
        metrics = trainer.train(local_batch(context.rank), learner_policy_version=0)

        parameter = next(model.module.parameters()).detach()
        gathered = [torch.empty_like(parameter) for _ in range(context.world_size)]
        torch.distributed.all_gather(gathered, parameter)
        if not all(torch.allclose(parameter, other, atol=1e-7, rtol=0.0) for other in gathered):
            raise AssertionError("DDP CausalLM replicas diverged after GRPO update")
        mean_loss = mean_across_ranks(torch.tensor(metrics["loss"], device=context.device))
        if context.is_primary:
            print(json.dumps({
                "world_size": context.world_size,
                "mean_loss": round(float(mean_loss.item()), 8),
                "replicas_synchronized": True,
                "trainer": "HuggingFaceTrainerWorker",
            }, sort_keys=True))
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
