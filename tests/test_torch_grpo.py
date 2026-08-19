import math
import unittest

from mini_verl.algorithms.grpo import grpo_loss_reference, torch_grpo_loss

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional dependency in the local development environment")
class TorchGrpoTest(unittest.TestCase):
    def test_matches_reference_and_backpropagates_only_through_valid_tokens(self):
        new = torch.tensor([[-0.1, -0.2], [-0.7, -0.4]], dtype=torch.float64, requires_grad=True)
        old = torch.tensor([[-0.2, -0.2], [-0.7, -0.3]], dtype=torch.float64)
        reference = torch.tensor([[-0.3, -0.5], [-0.8, -0.1]], dtype=torch.float64)
        advantage = torch.tensor([1.0, -0.5], dtype=torch.float64)
        mask = torch.tensor([[True, False], [True, True]])

        loss, metrics = torch_grpo_loss(
            new, old, advantage, mask, reference_logprobs=reference, clip_range=0.2, beta=0.07
        )
        reference_terms = grpo_loss_reference(
            new.detach().tolist(),
            old.tolist(),
            advantage.tolist(),
            mask.tolist(),
            reference_logprobs=reference.tolist(),
            clip_range=0.2,
            beta=0.07,
        )
        self.assertAlmostEqual(loss.item(), reference_terms.total_loss, places=12)
        self.assertAlmostEqual(metrics["policy_loss"].item(), reference_terms.policy_loss, places=12)
        self.assertAlmostEqual(metrics["kl_loss"].item(), reference_terms.kl_loss, places=12)
        self.assertAlmostEqual(metrics["mean_ratio"].item(), reference_terms.mean_ratio, places=12)
        self.assertAlmostEqual(metrics["clip_fraction"].item(), reference_terms.clip_fraction, places=12)
        self.assertEqual(metrics["token_count"].item(), reference_terms.token_count)

        loss.backward()
        self.assertIsNotNone(new.grad)
        self.assertAlmostEqual(new.grad[0, 1].item(), 0.0, places=12)
        self.assertNotEqual(new.grad[0, 0].item(), 0.0)

    def test_runs_on_cuda_when_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        device = torch.device("cuda:0")
        new = torch.tensor([[-0.1]], device=device, requires_grad=True)
        old = torch.tensor([[-0.1]], device=device)
        advantage = torch.tensor([1.0], device=device)
        mask = torch.tensor([[True]], device=device)
        loss, _ = torch_grpo_loss(new, old, advantage, mask)
        loss.backward()
        self.assertEqual(loss.device.type, "cuda")
        self.assertIsNotNone(new.grad)


if __name__ == "__main__":
    unittest.main()
