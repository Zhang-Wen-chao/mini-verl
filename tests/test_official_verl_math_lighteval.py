import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "official_verl" / "prepare_math_lighteval.py"
SPEC = importlib.util.spec_from_file_location("official_verl_math_lighteval", MODULE_PATH)
assert SPEC and SPEC.loader
math_lighteval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(math_lighteval)


class MathLightEvalContractTests(unittest.TestCase):
    def test_extracts_last_balanced_boxed_answer(self):
        slash = chr(92)
        self.assertEqual(
            math_lighteval.last_boxed_answer(f"first {slash}boxed{{1}}, final {slash}boxed{{{slash}frac{{1}}{{2}}}}"),
            f"{slash}frac{{1}}{{2}}",
        )
        with self.assertRaises(ValueError):
            math_lighteval.last_boxed_answer("no final answer")

    def test_row_preserves_the_math_rule_reward_contract(self):
        row = math_lighteval.official_row(
            {"problem": "What is two plus two?", "solution": f"It is {chr(92)}boxed{{4}}", "level": "Level 1", "type": "Algebra"},
            7,
        )
        self.assertEqual(row["data_source"], "DigitalLearningGmbH/MATH-lighteval")
        self.assertEqual(row["reward_model"]["ground_truth"], "4")
        self.assertEqual(row["extra_info"]["original_index"], 7)
        self.assertIn(f"{chr(92)}boxed{{...}}", row["prompt"][0]["content"])


if __name__ == "__main__":
    unittest.main()
