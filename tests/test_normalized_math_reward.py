import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "official_verl" / "compat"))
from normalized_math_reward import compute_score, exact_rational


class NormalizedMathRewardTest(unittest.TestCase):
    def test_decimal_and_fraction_are_equivalent(self):
        legacy = lambda output, target: 0.0
        self.assertEqual(compute_score(r"work \boxed{5.5}", r"\frac{11}{2}", legacy), 1.0)
        self.assertEqual(compute_score(r"work \boxed{3.5}", r"\frac{7}{2}", legacy), 1.0)

    def test_nested_boxed_fraction_is_equivalent(self):
        self.assertEqual(compute_score(r"\boxed{\dfrac{9}{7}}", r"\frac{9}{7}", lambda output, target: 0.0), 1.0)

    def test_non_equivalent_or_ambiguous_answers_stay_wrong(self):
        legacy = lambda output, target: 0.0
        self.assertEqual(compute_score(r"\boxed{-16}", "16", legacy), 0.0)
        self.assertEqual(compute_score(r"\boxed{12,10}", "12,10,6", legacy), 0.0)
        self.assertEqual(compute_score(r"\boxed{9}", "09", legacy), 0.0)
        self.assertEqual(compute_score("no boxed answer", r"\frac{11}{2}", legacy), 0.0)
        self.assertIsNone(exact_rational("x+1"))

    def test_last_boxed_answer_wins(self):
        self.assertEqual(
            compute_score(r"\boxed{1} then \boxed{5.5}", r"\frac{11}{2}", lambda output, target: 0.0),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
