import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional locally")
class CheckpointTest(unittest.TestCase):
    def test_restores_model_optimizer_policy_version_and_rng(self):
        from mini_verl.checkpoint import load_checkpoint, save_checkpoint

        torch.manual_seed(13)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        inputs = torch.tensor([[1.0, 2.0, 3.0]])
        model(inputs).sum().backward()
        optimizer.step()
        expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            saved = save_checkpoint(path, model=model, optimizer=optimizer, policy_version=8, extra={"epoch": 3})
            expected_next_random = torch.rand(4)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)

            restored = load_checkpoint(path, model=model, optimizer=optimizer)

        self.assertEqual(saved, restored)
        self.assertEqual(restored.policy_version, 8)
        self.assertEqual(restored.extra, {"epoch": 3})
        self.assertTrue(all(torch.equal(expected, actual) for expected, actual in zip(expected_parameters, model.parameters())))
        self.assertTrue(torch.equal(expected_next_random, torch.rand(4)))

    def test_rejects_invalid_policy_version(self):
        from mini_verl.checkpoint import save_checkpoint

        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "policy_version"):
                save_checkpoint(Path(directory) / "policy.pt", model=model, optimizer=optimizer, policy_version=-1)


if __name__ == "__main__":
    unittest.main()
