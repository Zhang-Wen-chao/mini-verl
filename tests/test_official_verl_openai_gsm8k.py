import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "official_verl" / "prepare_openai_gsm8k.py"
SPEC = importlib.util.spec_from_file_location("official_verl_openai_gsm8k", MODULE_PATH)
assert SPEC and SPEC.loader
gsm8k = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gsm8k)


class OpenAiGsm8kContractTests(unittest.TestCase):
    def test_extract_solution_matches_upstream_gsm8k_convention(self):
        self.assertEqual(gsm8k.extract_solution("reasoning\n#### 1,024"), "1024")
        self.assertEqual(gsm8k.extract_solution("reasoning\n#### -2.5"), "-2.5")
        with self.assertRaisesRegex(ValueError, "must end"):
            gsm8k.extract_solution("#### 42 and more")

    def test_converts_to_official_verl_rule_reward_schema(self):
        row = gsm8k.convert_record(
            {"question": "What is two plus two?", "answer": "Work.\n#### 4"},
            split="train",
            index=7,
        )
        self.assertEqual(row["data_source"], "openai/gsm8k")
        self.assertEqual(row["ability"], "math")
        self.assertEqual(row["reward_model"], {"style": "rule", "ground_truth": "4"})
        self.assertEqual(row["extra_info"]["index"], 7)
        self.assertIn("####", row["prompt"][0]["content"])

    def test_limit_preserves_original_row_indices(self):
        records = [
            {"question": "q0", "answer": "#### 0"},
            {"question": "q1", "answer": "#### 1"},
        ]
        rows = list(gsm8k.convert_records(records, split="test", limit=1))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["extra_info"]["index"], 0)


if __name__ == "__main__":
    unittest.main()
