import unittest

from mini_verl.protocol import Trajectory, TrajectoryBatch, TrajectoryValidationError

try:
    import torch
except ModuleNotFoundError:
    torch = None


def batch() -> TrajectoryBatch:
    return TrajectoryBatch(
        (
            Trajectory(
                prompt_token_ids=(1, 2),
                response_token_ids=(3, 4),
                old_logprobs=(-0.1, -0.2),
                response_mask=(True, False),
                policy_version=0,
                group_id="a",
            ),
            Trajectory(
                prompt_token_ids=(5,),
                response_token_ids=(2,),
                old_logprobs=(-0.3,),
                policy_version=0,
                group_id="b",
            ),
        )
    )


@unittest.skipIf(torch is None, "PyTorch is optional locally")
class ResponseLogprobTest(unittest.TestCase):
    def test_extracts_causal_response_logprobs_and_respects_masks(self):
        from mini_verl.tensors import response_logprobs_from_logits

        # Row 0: response (3, 4) is predicted by logits at positions (1, 2).
        # Row 1: response (2,) is predicted by logits at position 0.
        logits = torch.zeros((2, 4, 6), dtype=torch.float64)
        logits[0, 1, 3] = 2.0
        logits[0, 2, 4] = 3.0
        logits[1, 0, 2] = 4.0

        result = response_logprobs_from_logits(logits, batch())
        expected_first = torch.log_softmax(logits[0, 1], dim=-1)[3]
        expected_second = torch.log_softmax(logits[1, 0], dim=-1)[2]
        self.assertTrue(torch.equal(result.mask, torch.tensor([[True, False], [True, False]])))
        self.assertAlmostEqual(result.values[0, 0].item(), expected_first.item(), places=12)
        self.assertEqual(result.values[0, 1].item(), 0.0)
        self.assertAlmostEqual(result.values[1, 0].item(), expected_second.item(), places=12)
        self.assertEqual(result.values[1, 1].item(), 0.0)

    def test_rejects_logits_that_cannot_score_a_complete_response(self):
        from mini_verl.tensors import response_logprobs_from_logits

        with self.assertRaisesRegex(TrajectoryValidationError, "full prompt and response"):
            response_logprobs_from_logits(torch.zeros((2, 2, 6)), batch())

    def test_runs_on_cuda_when_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from mini_verl.tensors import response_logprobs_from_logits

        result = response_logprobs_from_logits(torch.zeros((2, 4, 6), device="cuda"), batch())
        self.assertEqual(result.values.device.type, "cuda")
        self.assertEqual(result.mask.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
