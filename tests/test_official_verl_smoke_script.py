import subprocess
import unittest
from pathlib import Path


OFFICIAL_VERL_DIR = Path(__file__).resolve().parents[1] / "official_verl"
SCRIPT = OFFICIAL_VERL_DIR / "run_qwen3_0_6b_4gpu_smoke.sh"
LOCAL_BOOTSTRAP = OFFICIAL_VERL_DIR / "bootstrap_local_official_env.sh"
WAIT_AND_RUN = OFFICIAL_VERL_DIR / "wait_and_run_qwen3_0_6b_4gpu_smoke.sh"


class OfficialVerlSmokeScriptTests(unittest.TestCase):
    def test_script_has_valid_bash_syntax_and_official_entrypoint(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("verl.experimental.one_step_off_policy.main_ppo", text)
        self.assertIn("algorithm.adv_estimator=grpo", text)
        self.assertIn("trainer.n_gpus_per_node=2", text)
        self.assertIn("rollout.n_gpus_per_node=2", text)
        self.assertIn("preflight.py", text)
        self.assertIn("--require-lock", text)
        self.assertIn("--require-runtime", text)
        self.assertLess(text.index("--require-runtime"), text.index("mkdir -p"))
        self.assertIn("PIPESTATUS[0]", text)
        self.assertIn("logs/exit_status", text)

    def test_local_bootstrap_uses_a_rebuildable_local_fsdp_vllm_environment(self):
        subprocess.run(["bash", "-n", str(LOCAL_BOOTSTRAP)], check=True)
        text = LOCAL_BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn("UV_PROJECT_ENVIRONMENT", text)
        self.assertIn("UV_CACHE_DIR", text)
        self.assertIn("uv sync --python 3.12 --extra fsdp --extra vllm", text)
        modules = '("torch", "transformers", "ray", "vllm", "verl")'
        self.assertIn(modules, text)
        runtime_check = text[text.index("for module in") :]
        self.assertNotIn('mbridge")', runtime_check)

    def test_wait_and_run_script_has_valid_bash_syntax_and_runtime_check(self):
        subprocess.run(["bash", "-n", str(WAIT_AND_RUN)], check=True)
        text = WAIT_AND_RUN.read_text(encoding="utf-8")
        self.assertIn("flock -n", text)
        self.assertIn('("torch", "transformers", "ray", "vllm", "verl")', text)
