import math
import unittest

from mini_verl.algorithms.grpo import grpo_loss_reference
from mini_verl.algorithms.ppo import generalized_advantage_estimate, ppo_loss_reference


class GeneralizedAdvantageEstimateTest(unittest.TestCase):
    def test_hand_calculated_two_step_trajectory(self):
        advantages, returns = generalized_advantage_estimate(
            rewards=(1.0, 2.0), values=(0.5, 1.0), dones=(False, True), gamma=1.0, gae_lambda=1.0
        )
        # delta_1 = 2 - 1 = 1; delta_0 = 1 + 1 - .5 = 1.5; A_0 = 1.5 + A_1.
        self.assertEqual(advantages, (2.5, 1.0))
        self.assertEqual(returns, (3.0, 2.0))

    def test_terminal_step_does_not_use_bootstrap_value(self):
        advantages, returns = generalized_advantage_estimate(
            rewards=(1.0,), values=(0.25,), dones=(True,), bootstrap_value=999.0
        )
        self.assertEqual(advantages, (0.75,))
        self.assertEqual(returns, (1.0,))

    def test_rejects_invalid_lengths_and_hyperparameters(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            generalized_advantage_estimate((1.0,), (0.0, 1.0), (True,))
        with self.assertRaisesRegex(ValueError, "gamma"):
            generalized_advantage_estimate((1.0,), (0.0,), (True,), gamma=1.1)


class PpoReferenceTest(unittest.TestCase):
    def test_actor_term_matches_grpo_when_advantages_are_fixed(self):
        new = (math.log(1.5), math.log(0.5))
        old = (0.0, 0.0)
        advantages = (1.0, -1.0)
        ppo = ppo_loss_reference(new, old, advantages, new_values=(0.0, 0.0), returns=(0.0, 0.0))
        grpo = grpo_loss_reference(
            new_logprobs=((new[0],), (new[1],)),
            old_logprobs=((old[0],), (old[1],)),
            advantages=advantages,
            response_mask=((True,), (True,)),
        )
        self.assertAlmostEqual(ppo.actor_loss, grpo.policy_loss)
        self.assertEqual(ppo.value_loss, 0.0)

    def test_clip_and_critic_losses_have_expected_values(self):
        terms = ppo_loss_reference(
            new_logprobs=(math.log(1.5), math.log(0.5)),
            old_logprobs=(0.0, 0.0),
            advantages=(1.0, -1.0),
            new_values=(0.0, 2.0),
            returns=(1.0, 0.0),
            clip_range=0.2,
            value_coef=0.5,
        )
        # Actor: mean(-1.2, +0.8) = -0.2. Critic: mean(.5, 2.0) = 1.25.
        self.assertAlmostEqual(terms.actor_loss, -0.2)
        self.assertAlmostEqual(terms.value_loss, 1.25)
        self.assertAlmostEqual(terms.total_loss, 0.425)
        self.assertAlmostEqual(terms.clip_fraction, 1.0)

    def test_advantage_source_is_not_group_normalization(self):
        advantages, returns = generalized_advantage_estimate(
            rewards=(0.0, 0.0, 1.0, 1.0),
            values=(0.25, 0.25, 0.25, 0.25),
            dones=(True, True, True, True),
        )
        self.assertEqual(advantages, (-0.25, -0.25, 0.75, 0.75))
        self.assertEqual(returns, (0.0, 0.0, 1.0, 1.0))

    def test_rejects_incompatible_inputs(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            ppo_loss_reference((0.0,), (0.0, 1.0), (1.0,), (0.0,), (1.0,))


if __name__ == "__main__":
    unittest.main()

