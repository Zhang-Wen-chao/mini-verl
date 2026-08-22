import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "official_verl" / "gsm8k.py"
SPEC = importlib.util.spec_from_file_location("official_verl_gsm8k", MODULE_PATH)
assert SPEC and SPEC.loader
gsm8k = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gsm8k)


class Gsm8kContractTests(unittest.TestCase):
    def test_converts_final_marker_and_builds_prompt(self):
        rows = list(gsm8k.convert_records([{"question": "2 plus 3?", "answer": "work\n#### 5"}]))
        self.assertEqual(rows[0]["answer"], "5")
        self.assertIn("\\boxed{integer}", rows[0]["prompt"])

    def test_rejects_missing_or_non_integer_gsm8k_answer(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            gsm8k.gsm8k_final_answer("no marker")
        with self.assertRaisesRegex(ValueError, "integer"):
            gsm8k.gsm8k_final_answer("#### 5.0")

    def test_reward_requires_last_token_to_be_boxed_integer(self):
        self.assertEqual(gsm8k.exact_boxed_integer_reward("reasoning \\boxed{1,024}", "1024"), 1.0)
        self.assertEqual(gsm8k.exact_boxed_integer_reward("\\boxed{1024} extra", "1024"), 0.0)
        self.assertEqual(gsm8k.exact_boxed_integer_reward("\\boxed{1023}", "1024"), 0.0)

    def test_jsonl_round_trip_honors_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "rows.jsonl"
            count = gsm8k.write_jsonl(iter([{"prompt": "a", "answer": "1"}, {"prompt": "b", "answer": "2"}]), output, limit=1)
            self.assertEqual(count, 1)
            self.assertEqual(list(gsm8k.read_jsonl(output)), [{"prompt": "a", "answer": "1"}])


if __name__ == "__main__":
    unittest.main()
