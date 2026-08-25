import unittest

from answer_protocol import extract_final_answer, scoreable_answer_text


class AnswerProtocolTests(unittest.TestCase):
    def test_explicit_boxed_answer_is_extracted(self):
        self.assertEqual(extract_final_answer("Working...\nAnswer: \\boxed{42}"), "42")

    def test_bare_boxed_answer_is_accepted(self):
        self.assertEqual(extract_final_answer("Thus the result is \\boxed{699}."), "699")

    def test_recovery_placeholder_is_not_an_answer(self):
        self.assertIsNone(extract_final_answer("Submit \\boxed{answer} when done."))

    def test_last_real_answer_survives_a_later_placeholder(self):
        text = "Answer: \\boxed{601}\nRecovery: submit \\boxed{answer}."
        self.assertEqual(extract_final_answer(text), "601")

    def test_tool_call_comment_is_not_a_final_answer(self):
        text = '<tool_call>{"name":"code_interpreter","arguments":{"code":"# Answer: \\boxed{999}"}}</tool_call>'
        self.assertIsNone(extract_final_answer(text))

    def test_only_the_extracted_answer_is_scored(self):
        self.assertEqual(scoreable_answer_text("73"), "Answer: \\boxed{73}")
        self.assertEqual(scoreable_answer_text(None), "")


if __name__ == "__main__":
    unittest.main()
