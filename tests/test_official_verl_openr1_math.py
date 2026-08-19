import importlib.util
import unittest
from collections import Counter
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "official_verl" / "prepare_openr1_math.py"
SPEC = importlib.util.spec_from_file_location("official_verl_openr1_math", MODULE_PATH)
assert SPEC and SPEC.loader
openr1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(openr1)


class OpenR1MathContractTests(unittest.TestCase):
    def test_problem_normalization_is_conservative_and_deterministic(self):
        self.assertEqual(openr1.normalize_problem("  Solve  X  \n"), "solve x")
        self.assertNotEqual(openr1.normalize_problem("x+1"), openr1.normalize_problem("x+2"))
        self.assertEqual(openr1.stable_rank(42, "id", "x"), openr1.stable_rank(42, "id", "x"))

    def test_filters_explicit_source_and_exact_evaluation_duplicates(self):
        valid = {"problem": "Unique question", "answer": "1", "source": "olympiads"}
        self.assertEqual(openr1.is_eligible(valid, set()), (True, None))
        self.assertEqual(openr1.is_eligible({**valid, "source": "MATH"}, set()), (False, "blocked_source"))
        self.assertEqual(openr1.is_eligible(valid, {"unique question"}), (False, "evaluation_exact_duplicate"))

    def test_proportional_quotas_sum_to_requested_limit(self):
        quotas = openr1.proportional_quotas(Counter({"Algebra": 8, "Geometry": 2}), 5)
        self.assertEqual(sum(quotas.values()), 5)
        self.assertEqual(quotas, {"Algebra": 4, "Geometry": 1})

    def test_official_row_uses_boxed_contract_and_raw_answer(self):
        row = openr1.official_verl_row(
            {
                "problem": "What is two plus two?",
                "answer": "4",
                "source": "olympiads",
                "problem_type": "Algebra",
                "question_type": "short-answer",
                "uuid": "u-1",
            },
            split="train",
            original_index=7,
        )
        self.assertEqual(row["data_source"], "DigitalLearningGmbH/MATH-lighteval")
        self.assertEqual(row["reward_model"], {"style": "rule", "ground_truth": "4"})
        self.assertEqual(row["extra_info"]["source"], "olympiads")
        self.assertEqual(row["extra_info"]["uuid"], "u-1")
        self.assertEqual(row["extra_info"]["raw_data_source"], "open-r1/OpenR1-Math-220k")
        self.assertIn("\\boxed{...}", row["prompt"][0]["content"])
        self.assertIn("at most three short sentences", row["prompt"][0]["content"])


if __name__ == "__main__":
    unittest.main()
