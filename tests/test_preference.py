import unittest

from mini_verl.preference import PreferencePair, preference_pairs
from mini_verl.protocol import Trajectory, TrajectoryBatch, TrajectoryValidationError


def _trajectory(group_id: str, reward: float | None, response: tuple[int, ...] = (3,)) -> Trajectory:
    return Trajectory(
        prompt_token_ids=(1, 2),
        response_token_ids=response,
        old_logprobs=tuple(-2.0 for _ in response),
        policy_version=0,
        group_id=group_id,
        reward=reward,
    )


class PreferencePairsTest(unittest.TestCase):
    def test_pairs_pick_highest_and_lowest_reward_within_each_group(self):
        batch = TrajectoryBatch((
            _trajectory("a", 1.0, response=(3, 4)),
            _trajectory("a", 0.0, response=(5,)),
            _trajectory("b", 0.5, response=(6,)),
            _trajectory("b", 1.0, response=(7, 8)),
            _trajectory("b", 0.0, response=(9,)),
        ))
        pairs = preference_pairs(batch)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].group_id, "a")
        self.assertEqual(pairs[0].chosen.response_token_ids, (3, 4))
        self.assertEqual(pairs[0].rejected.response_token_ids, (5,))
        self.assertEqual(pairs[1].group_id, "b")
        self.assertEqual(pairs[1].chosen.response_token_ids, (7, 8))
        self.assertEqual(pairs[1].rejected.response_token_ids, (9,))
        self.assertEqual([pair.reward_margin for pair in pairs], [1.0, 1.0])

    def test_pairs_are_preference_pair_instances_with_scored_trajectories(self):
        batch = TrajectoryBatch((_trajectory("a", 1.0), _trajectory("a", 0.0)))
        pair = preference_pairs(batch)[0]
        self.assertIsInstance(pair, PreferencePair)
        self.assertEqual(pair.chosen.reward, 1.0)
        self.assertEqual(pair.rejected.reward, 0.0)

    def test_tied_group_produces_no_pair(self):
        batch = TrajectoryBatch((_trajectory("a", 1.0), _trajectory("a", 1.0)))
        self.assertEqual(preference_pairs(batch), ())

    def test_min_reward_margin_filters_small_spreads(self):
        batch = TrajectoryBatch((_trajectory("a", 1.0), _trajectory("a", 0.0)))
        self.assertEqual(preference_pairs(batch, min_reward_margin=1.0), ())
        self.assertEqual(len(preference_pairs(batch, min_reward_margin=0.99)), 1)

    def test_missing_rewards_are_rejected(self):
        batch = TrajectoryBatch((_trajectory("a", 1.0), _trajectory("a", None)))
        with self.assertRaisesRegex(TrajectoryValidationError, "need a reward"):
            preference_pairs(batch)

    def test_single_trajectory_group_is_rejected(self):
        batch = TrajectoryBatch((_trajectory("a", 1.0),))
        with self.assertRaisesRegex(TrajectoryValidationError, "at least two trajectories"):
            preference_pairs(batch)

    def test_negative_min_reward_margin_is_rejected(self):
        batch = TrajectoryBatch((_trajectory("a", 1.0), _trajectory("a", 0.0)))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            preference_pairs(batch, min_reward_margin=-0.1)


if __name__ == "__main__":
    unittest.main()
