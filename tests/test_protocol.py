import unittest

from mini_verl.protocol import Trajectory, TrajectoryBatch, TrajectoryValidationError


def make_trajectory(**overrides):
    values = {
        "prompt_token_ids": (1, 2),
        "response_token_ids": (3, 4),
        "old_logprobs": (-0.2, -0.4),
        "policy_version": 7,
        "group_id": "prompt-1",
    }
    values.update(overrides)
    return Trajectory(**values)


class TrajectoryTest(unittest.TestCase):
    def test_defaults_make_all_response_tokens_trainable(self):
        trajectory = make_trajectory()
        self.assertEqual(trajectory.response_mask, (True, True))
        self.assertEqual(trajectory.response_token_count, 2)

    def test_rejects_logprob_length_mismatch(self):
        with self.assertRaisesRegex(TrajectoryValidationError, "old_logprobs"):
            make_trajectory(old_logprobs=(-0.2,))

    def test_rejects_an_all_padding_response(self):
        with self.assertRaisesRegex(TrajectoryValidationError, "trainable token"):
            make_trajectory(response_mask=(False, False))

    def test_round_trip_preserves_transport_contract(self):
        trajectory = make_trajectory(
            response_mask=(True, False),
            reference_logprobs=(-0.3, -0.5),
            reward=1.0,
            advantage=0.9,
            metadata={"expected_answer": "42"},
        )
        restored = Trajectory.from_dict(trajectory.to_dict())
        self.assertEqual(restored, trajectory)


class TrajectoryBatchTest(unittest.TestCase):
    def test_policy_version_guard_rejects_stale_trajectories(self):
        batch = TrajectoryBatch((make_trajectory(policy_version=7), make_trajectory(policy_version=6)))
        with self.assertRaisesRegex(TrajectoryValidationError, "stale"):
            batch.require_policy_version(7)

    def test_groups_and_token_count(self):
        batch = TrajectoryBatch(
            (
                make_trajectory(group_id="a", response_mask=(True, False)),
                make_trajectory(group_id="a"),
                make_trajectory(group_id="b"),
            )
        )
        self.assertEqual(set(batch.groups()), {"a", "b"})
        self.assertEqual(batch.response_token_count, 5)


if __name__ == "__main__":
    unittest.main()
