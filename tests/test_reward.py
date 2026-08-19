import unittest

from mini_verl.protocol import Trajectory, TrajectoryBatch, TrajectoryValidationError
from mini_verl.reward import apply_rewards, exact_answer_reward, group_relative_advantages


def sample(*, group_id="prompt", answer="42", expected="42") -> Trajectory:
    return Trajectory(
        prompt_token_ids=(1,),
        response_token_ids=(2,),
        old_logprobs=(-0.1,),
        policy_version=0,
        group_id=group_id,
        response_text=f"work\nFinal answer: {answer}",
        metadata={"expected_answer": expected},
    )


class RewardTest(unittest.TestCase):
    def test_exact_answer_uses_final_answer_line(self):
        self.assertEqual(exact_answer_reward(sample(answer="42")), 1.0)
        self.assertEqual(exact_answer_reward(sample(answer="41")), 0.0)

    def test_group_advantage_is_invariant_to_reward_shift(self):
        original = TrajectoryBatch((sample(), sample(answer="41")))
        scored = apply_rewards(original, exact_answer_reward)
        shifted = TrajectoryBatch(tuple(trajectory.with_reward(trajectory.reward + 11) for trajectory in scored.trajectories))

        original_advantages = [item.advantage for item in group_relative_advantages(scored).trajectories]
        shifted_advantages = [item.advantage for item in group_relative_advantages(shifted).trajectories]
        self.assertEqual(original_advantages, shifted_advantages)

    def test_constant_reward_group_has_zero_advantages(self):
        scored = TrajectoryBatch((sample(), sample(answer="42"))).to_dict()
        batch = TrajectoryBatch.from_dict(scored)
        result = group_relative_advantages(apply_rewards(batch, exact_answer_reward))
        self.assertEqual([item.advantage for item in result.trajectories], [0.0, 0.0])

    def test_advantage_requires_rewards(self):
        with self.assertRaisesRegex(TrajectoryValidationError, "need a reward"):
            group_relative_advantages(TrajectoryBatch((sample(),)))


if __name__ == "__main__":
    unittest.main()
