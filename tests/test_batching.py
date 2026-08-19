import unittest

from mini_verl.batching import length_bucket_batches
from mini_verl.protocol import Trajectory


def trajectory(index: int, *, prompt_length: int, response_length: int) -> Trajectory:
    return Trajectory(
        prompt_token_ids=tuple(range(1, prompt_length + 1)),
        response_token_ids=tuple(range(1, response_length + 1)),
        old_logprobs=tuple(-0.1 for _ in range(response_length)),
        policy_version=0,
        group_id=f"group-{index}",
    )


class LengthBucketTest(unittest.TestCase):
    def test_batches_are_length_sorted_and_respect_budgets(self):
        items = (
            trajectory(0, prompt_length=2, response_length=1),
            trajectory(1, prompt_length=2, response_length=8),
            trajectory(2, prompt_length=2, response_length=2),
            trajectory(3, prompt_length=2, response_length=7),
        )
        batches = length_bucket_batches(items, max_batch_size=2, max_padded_tokens=20)

        self.assertEqual([[len(item.response_token_ids) for item in packed.batch.trajectories] for packed in batches], [[8, 7], [2, 1]])
        self.assertTrue(all(len(packed.batch.trajectories) <= 2 for packed in batches))
        self.assertTrue(all(packed.padded_sequence_tokens <= 20 for packed in batches))
        self.assertEqual(sum(len(packed.batch.trajectories) for packed in batches), len(items))

    def test_reports_padding_accounting(self):
        batches = length_bucket_batches(
            (trajectory(0, prompt_length=2, response_length=4), trajectory(1, prompt_length=2, response_length=1)),
            max_batch_size=2,
            max_padded_tokens=20,
        )
        packed = batches[0]
        self.assertEqual(packed.real_sequence_tokens, 9)
        self.assertEqual(packed.padded_sequence_tokens, 12)
        self.assertEqual(packed.padding_tokens, 3)
        self.assertAlmostEqual(packed.padding_ratio, 0.25)

    def test_rejects_a_trajectory_larger_than_budget(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            length_bucket_batches((trajectory(0, prompt_length=2, response_length=8),), max_batch_size=2, max_padded_tokens=8)


if __name__ == "__main__":
    unittest.main()
