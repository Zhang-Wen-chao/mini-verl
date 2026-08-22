import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional dependency in the local development environment")
class ToyPpoIntegrationTest(unittest.TestCase):
    def test_same_categorical_pass_at_1_metric_improves(self):
        from examples.toy_ppo_train import run

        result = run(device="cpu", iterations=60, seed=7)
        self.assertGreater(result.final_pass_at_1, result.initial_pass_at_1)


if __name__ == "__main__":
    unittest.main()

