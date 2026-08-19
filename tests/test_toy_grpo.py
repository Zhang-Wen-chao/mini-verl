import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional locally")
class ToyGrpoIntegrationTest(unittest.TestCase):
    def test_end_to_end_loop_improves_policy(self):
        from mini_verl import run_toy_grpo

        result = run_toy_grpo(device="cpu", iterations=70, seed=7)
        self.assertGreater(result.final_pass_at_1, result.initial_pass_at_1)
        self.assertGreaterEqual(result.final_pass_at_1, 0.875)
        self.assertEqual(result.completed_iterations, 70)

    def test_rejects_a_group_that_cannot_produce_relative_advantage(self):
        from mini_verl import run_toy_grpo

        with self.assertRaisesRegex(ValueError, "group_size"):
            run_toy_grpo(device="cpu", iterations=1, group_size=1)


if __name__ == "__main__":
    unittest.main()
