"""CPU-only preflight for the standalone PPO Critic model config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppo_critic_config import FSDPCriticHFModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()

    model_path = args.model.resolve()
    config = FSDPCriticHFModelConfig(path=str(model_path), tokenizer_path=str(model_path))
    if config.hf_config is None or config.generation_config is None:
        raise SystemExit("HF model metadata was not initialized")
    if not config.local_path or config.get_processor() is None:
        raise SystemExit("local model path or tokenizer/processor was not initialized")

    print(
        json.dumps(
            {
                "config_type": type(config).__name__,
                "hf_config_type": type(config.hf_config).__name__,
                "local_path": config.local_path,
                "architectures": config.architectures,
                "processor_type": type(config.get_processor()).__name__,
                "fsdp_strategy": config.fsdp_config.strategy,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
