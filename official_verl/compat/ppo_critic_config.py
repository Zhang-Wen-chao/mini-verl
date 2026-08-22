"""PPO Critic model configuration compatible with the pinned VeRL engine.

The pinned experimental one-step trainer expects ``model.fsdp_config`` for a
standalone Critic, while the in-tree ``FSDPCriticModelCfg`` only inherits
``BaseModelConfig``. The FSDP engine, however, requires the Hugging Face
metadata initialized by ``HFModelConfig.__post_init__`` (``hf_config``,
``local_path``, tokenizer/processor, and generation config).

This adapter retains the full supported HF-model initialization contract and
adds only the standalone Critic's FSDP engine block. It is imported by Hydra
after VeRL starts normally; ``sitecustomize.py`` remains import-safe.
"""

from dataclasses import dataclass, field

from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.config.model import HFModelConfig


@dataclass
class FSDPCriticHFModelConfig(HFModelConfig):
    """An initialized HF model config plus the Critic's FSDP options."""

    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
