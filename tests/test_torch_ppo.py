import unittest

from mini_verl.algorithms.ppo import ppo_loss_reference, torch_ppo_loss

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional dependency in the local development environment")
class TorchPpoTest(unittest.TestCase):
    def test_matches_reference_and_backpropagates_to_actor_and_critic(self):
        new_logprobs = torch.tensor([-0.1, -0.8], dtype=torch.float64, requires_grad=True)
        old_logprobs = torch.tensor([-0.2, -0.3], dtype=torch.float64)
        advantages = torch.tensor([1.0, -0.5], dtype=torch.float64)
        new_values = torch.tensor([0.2, 0.9], dtype=torch.float64, requires_grad=True)
        returns = torch.tensor([1.0, 0.0], dtype=torch.float64)
        reference = torch.tensor([-0.3, -0.4], dtype=torch.float64)

        loss, metrics = torch_ppo_loss(
            new_logprobs, old_logprobs, advantages, new_values, returns,
            reference_logprobs=reference, clip_range=0.2, value_coef=0.5, beta=0.07
        )
        expected = ppo_loss_reference(
            new_logprobs.detach().tolist(), old_logprobs.tolist(), advantages.tolist(),
            new_values.detach().tolist(), returns.tolist(), reference_logprobs=reference.tolist(),
            clip_range=0.2, value_coef=0.5, beta=0.07
        )
        self.assertAlmostEqual(loss.item(), expected.total_loss, places=12)
        self.assertAlmostEqual(metrics["actor_loss"].item(), expected.actor_loss, places=12)
        self.assertAlmostEqual(metrics["value_loss"].item(), expected.value_loss, places=12)
        self.assertAlmostEqual(metrics["kl_loss"].item(), expected.kl_loss, places=12)

        loss.backward()
        self.assertNotEqual(new_logprobs.grad[0].item(), 0.0)
        self.assertNotEqual(new_values.grad[0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()

