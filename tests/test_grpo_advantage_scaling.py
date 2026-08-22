import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "official_verl" / "analyze_grpo_advantage_scaling.py"
SPEC = importlib.util.spec_from_file_location("grpo_advantage_scaling", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GrpoAdvantageScalingTest(unittest.TestCase):
    def test_binary_group_of_four_matches_pinned_grpo_scale(self):
        centered = MODULE.advantages([1.0, 0.0, 0.0, 0.0], normalize_by_std=False)
        standardized = MODULE.advantages([1.0, 0.0, 0.0, 0.0], normalize_by_std=True)
        self.assertEqual(centered, [0.75, -0.25, -0.25, -0.25])
        self.assertAlmostEqual(standardized[0], 1.5, places=5)
        self.assertAlmostEqual(standardized[1], -0.5, places=5)

    def test_degenerate_group_remains_zero(self):
        self.assertEqual(MODULE.advantages([0.0, 0.0, 0.0, 0.0], normalize_by_std=True), [0.0] * 4)


if __name__ == "__main__":
    unittest.main()
