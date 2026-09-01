import unittest

from mini_verl.algorithms.dpo import dpo_loss_reference, torch_dpo_loss

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional dependency in the local development environment")
class TorchDpoTest(unittest.TestCase):
    def _inputs(self):
        new = torch.tensor(
            [
                [0.0, -1.0, 5.0],
                [1.0, 0.0, 5.0],
                [-2.0, -2.0, 5.0],
                [0.0, 0.0, 5.0],
            ],
            dtype=torch.float64,
        )
        reference = torch.tensor(
            [
                [-2.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [-2.0, -1.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        mask = torch.tensor(
            [
                [True, True, False],
                [True, True, False],
                [True, False, False],
                [True, True, False],
            ]
        )
        return new, reference, mask

    def test_matches_reference_and_backpropagates_only_through_valid_tokens(self):
        new, reference, mask = self._inputs()
        new = new.detach().requires_grad_(True)
        loss, metrics = torch_dpo_loss(new, reference, mask, beta=0.1)
        reference_terms = dpo_loss_reference(
            new.detach().tolist(), reference.tolist(), mask.tolist(), beta=0.1
        )

        self.assertAlmostEqual(loss.item(), reference_terms.total_loss, places=12)
        self.assertAlmostEqual(metrics["chosen_reward"].item(), reference_terms.chosen_reward, places=12)
        self.assertAlmostEqual(metrics["rejected_reward"].item(), reference_terms.rejected_reward, places=12)
        self.assertAlmostEqual(metrics["reward_margin"].item(), reference_terms.reward_margin, places=12)
        self.assertAlmostEqual(metrics["accuracy"].item(), reference_terms.accuracy, places=12)
        self.assertEqual(metrics["pair_count"].item(), reference_terms.pair_count)

        loss.backward()
        for row, mask_row in enumerate(mask.tolist()):
            for column, valid in enumerate(mask_row):
                gradient = new.grad[row, column].item()
                if valid:
                    self.assertNotEqual(gradient, 0.0)
                else:
                    self.assertEqual(gradient, 0.0)

    def test_matches_reference_with_length_normalization(self):
        new, reference, mask = self._inputs()
        new = new.detach().requires_grad_(True)
        loss, metrics = torch_dpo_loss(new, reference, mask, beta=0.1, length_normalize=True)
        reference_terms = dpo_loss_reference(
            new.detach().tolist(), reference.tolist(), mask.tolist(), beta=0.1, length_normalize=True
        )

        self.assertAlmostEqual(loss.item(), reference_terms.total_loss, places=12)
        self.assertAlmostEqual(metrics["chosen_reward"].item(), reference_terms.chosen_reward, places=12)
        self.assertAlmostEqual(metrics["rejected_reward"].item(), reference_terms.rejected_reward, places=12)
        self.assertAlmostEqual(metrics["reward_margin"].item(), reference_terms.reward_margin, places=12)
        self.assertAlmostEqual(metrics["accuracy"].item(), reference_terms.accuracy, places=12)

    def test_rejects_fully_masked_row(self):
        new, reference, mask = self._inputs()
        mask[2] = torch.tensor([False, False, False])
        with self.assertRaisesRegex(ValueError, "at least one valid token in every row"):
            torch_dpo_loss(new, reference, mask)

    def test_runs_on_cuda_when_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available in this environment")
        new, reference, mask = self._inputs()
        new = new.detach().to("cuda").requires_grad_(True)
        reference = reference.to("cuda")
        mask = mask.to("cuda")
        loss, metrics = torch_dpo_loss(new, reference, mask, beta=0.1)
        reference_terms = dpo_loss_reference(
            new.detach().cpu().tolist(), reference.cpu().tolist(), mask.cpu().tolist(), beta=0.1
        )
        self.assertAlmostEqual(loss.item(), reference_terms.total_loss, places=12)
        self.assertAlmostEqual(metrics["accuracy"].item(), reference_terms.accuracy, places=12)


from mini_verl.algorithms.dpo import torch_dpo_loss


if __name__ == "__main__":
    unittest.main()
