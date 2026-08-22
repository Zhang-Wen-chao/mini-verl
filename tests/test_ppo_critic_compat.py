import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPAT = ROOT / "official_verl" / "compat" / "ppo_critic_config.py"
VERIFY = ROOT / "official_verl" / "compat" / "verify_ppo_critic_config.py"


class PPOCriticCompatTests(unittest.TestCase):
    def test_adapter_keeps_hf_initialization_contract(self):
        source = COMPAT.read_text(encoding="utf-8")
        self.assertIn("class FSDPCriticHFModelConfig(HFModelConfig)", source)
        self.assertIn("fsdp_config: FSDPEngineConfig", source)
        self.assertNotIn("def __post_init__", source)

    def test_cpu_preflight_checks_real_hf_metadata(self):
        source = VERIFY.read_text(encoding="utf-8")
        self.assertIn("FSDPCriticHFModelConfig", source)
        self.assertIn("config.hf_config is None", source)
        self.assertIn("config.generation_config is None", source)
        self.assertIn("config.get_processor() is None", source)


if __name__ == "__main__":
    unittest.main()
