import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional locally")
class ToyGrpoIntegrationTest(unittest.TestCase):
    def test_end_to_end_loop_improves_policy(self):
        from examples.toy_grpo_train import run

        result = run(device="cpu", iterations=70, seed=7)
        self.assertGreater(result.final_pass_at_1, result.initial_pass_at_1)
        self.assertGreaterEqual(result.final_pass_at_1, 0.875)


if __name__ == "__main__":
    unittest.main()
