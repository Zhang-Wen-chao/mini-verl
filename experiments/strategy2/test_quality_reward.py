import unittest

from quality_reward import code_fingerprint, normalize_markdown_code, quality_process_score, tool_execution_succeeded


class QualityRewardTests(unittest.TestCase):
    def test_empty_tool_output_is_a_success(self):
        self.assertTrue(tool_execution_succeeded(""))
        self.assertTrue(tool_execution_succeeded("Output:\n42"))

    def test_sandbox_and_runtime_errors_are_failures(self):
        self.assertFalse(tool_execution_succeeded("Error: Import of 'os' is not allowed"))
        self.assertFalse(tool_execution_succeeded("Traceback: boom"))

    def test_whitespace_only_difference_is_a_repeat(self):
        self.assertEqual(code_fingerprint("print( 1 )\n"), code_fingerprint("print( 1 )"))

    def test_only_outer_python_fence_is_normalized(self):
        self.assertEqual(normalize_markdown_code("```py\nprint(42)\n```"), ("print(42)", True))
        self.assertEqual(normalize_markdown_code("print('```py')"), ("print('```py')", False))

    def test_verified_answer_dominates_process_only_rollout(self):
        correct = quality_process_score(1.0, submitted_answer=True, tool_successes=1, tool_failures=0, invalid_actions=0, repeated_tool_calls=0)
        incorrect = quality_process_score(-1.0, submitted_answer=True, tool_successes=2, tool_failures=0, invalid_actions=0, repeated_tool_calls=0)
        self.assertGreater(correct, incorrect)
        self.assertLess(incorrect, 0)

    def test_bad_or_unfinished_rollout_is_penalized(self):
        useful = quality_process_score(-1.0, submitted_answer=True, tool_successes=1, tool_failures=0, invalid_actions=0, repeated_tool_calls=0)
        bad = quality_process_score(-1.0, submitted_answer=False, tool_successes=0, tool_failures=1, invalid_actions=1, repeated_tool_calls=1)
        self.assertGreater(useful, bad)

    def test_fenced_tool_call_is_less_preferred_than_clean_one(self):
        clean = quality_process_score(-1.0, submitted_answer=True, tool_successes=1, tool_failures=0, invalid_actions=0, repeated_tool_calls=0)
        fenced = quality_process_score(-1.0, submitted_answer=True, tool_successes=1, tool_failures=0, invalid_actions=0, repeated_tool_calls=0, markdown_fenced_tool_calls=1)
        self.assertGreater(clean, fenced)

    def test_fenced_tool_call_is_less_preferred_even_when_correct(self):
        clean = quality_process_score(1.0, submitted_answer=True, tool_successes=1, tool_failures=0, invalid_actions=0, repeated_tool_calls=0)
        fenced = quality_process_score(1.0, submitted_answer=True, tool_successes=1, tool_failures=0, invalid_actions=0, repeated_tool_calls=0, markdown_fenced_tool_calls=1)
        self.assertGreater(clean, fenced)

    def test_correct_answer_does_not_forgive_failed_or_repeated_calls(self):
        clean = quality_process_score(
            1.0, submitted_answer=True, tool_successes=1, tool_failures=0, invalid_actions=0, repeated_tool_calls=0
        )
        noisy = quality_process_score(
            1.0, submitted_answer=True, tool_successes=1, tool_failures=1, invalid_actions=1, repeated_tool_calls=1
        )
        self.assertGreater(clean, noisy)
        self.assertGreater(noisy, 0.0)


if __name__ == "__main__":
    unittest.main()
