import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "official_verl" / "summarize_grpo_development_pair.py"
SPEC = importlib.util.spec_from_file_location("grpo_pair_summary", MODULE_PATH)
assert SPEC and SPEC.loader
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


class GrpoDevelopmentPairSummaryTests(unittest.TestCase):
    def test_summarize_completed_run_and_render_development_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "checkpoints" / "global_step_2").mkdir(parents=True)
            (root / "rollout_samples").mkdir()
            for index in (1, 2):
                (root / "rollout_samples" / f"{index}.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "logs" / "exit_status").write_text("0\n", encoding="utf-8")
            (root / "logs" / "watch_exit_status").write_text("0\n", encoding="utf-8")
            (root / "logs" / "train.log").write_text(
                "step:0 - val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1:np.float64(0.5)\n"
                "step:1 - training/global_step:1 - actor/ppo_kl:np.float64(0.1) - "
                "critic/score/mean:0.5 - actor/grad_norm:np.float64(1.0) - timing_s/step:2.0\n"
                "step:2 - training/global_step:2 - actor/ppo_kl:np.float64(0.2) - "
                "critic/score/mean:0.0 - actor/grad_norm:np.float64(0.0) - timing_s/step:4.0\n"
                "step:2 - val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1:np.float64(0.75)\n",
                encoding="utf-8",
            )
            run = summary.summarize("test", root, 2)
            self.assertEqual(run.rollout_count, 2)
            self.assertEqual(run.finite_kl_steps, 2)
            self.assertEqual(run.mixed_nonzero_grad_groups, 1)
            self.assertAlmostEqual(run.mean_step_seconds or 0, 3.0)
            report = summary.render(run, run)
            self.assertIn("single-seed development comparison", report)
            self.assertIn("held-out 200 set was not read", report)


if __name__ == "__main__":
    unittest.main()
