import math
import unittest

from mini_verl.algorithms.grpo import grpo_loss_reference


class GrpoReferenceTest(unittest.TestCase):
    def test_same_policy_has_ratio_one_and_expected_policy_loss(self):
        terms = grpo_loss_reference(
            new_logprobs=((math.log(0.5), math.log(0.5)),),
            old_logprobs=((math.log(0.5), math.log(0.5)),),
            advantages=(2.0,),
            response_mask=((True, True),),
        )
        self.assertAlmostEqual(terms.mean_ratio, 1.0)
        self.assertAlmostEqual(terms.policy_loss, -2.0)
        self.assertEqual(terms.clip_fraction, 0.0)

    def test_positive_advantage_is_clipped_when_ratio_is_too_large(self):
        terms = grpo_loss_reference(
            new_logprobs=((math.log(1.5),),),
            old_logprobs=((0.0,),),
            advantages=(1.0,),
            response_mask=((True,),),
            clip_range=0.2,
        )
        self.assertAlmostEqual(terms.policy_loss, -1.2)
        self.assertAlmostEqual(terms.clip_fraction, 1.0)

    def test_negative_advantage_is_clipped_in_the_opposite_direction(self):
        terms = grpo_loss_reference(
            new_logprobs=((math.log(0.5),),),
            old_logprobs=((0.0,),),
            advantages=(-1.0,),
            response_mask=((True,),),
            clip_range=0.2,
        )
        self.assertAlmostEqual(terms.policy_loss, 0.8)

    def test_padding_token_does_not_change_loss_or_metrics(self):
        padded = grpo_loss_reference(
            new_logprobs=((math.log(1.5), math.log(100.0)),),
            old_logprobs=((0.0, 0.0),),
            advantages=(1.0,),
            response_mask=((True, False),),
            reference_logprobs=((math.log(1.5), 0.0),),
            beta=0.3,
        )
        unpadded = grpo_loss_reference(
            new_logprobs=((math.log(1.5),),),
            old_logprobs=((0.0,),),
            advantages=(1.0,),
            response_mask=((True,),),
            reference_logprobs=((math.log(1.5),),),
            beta=0.3,
        )
        self.assertEqual(padded, unpadded)

    def test_matching_reference_has_zero_kl(self):
        terms = grpo_loss_reference(
            new_logprobs=((-0.4,),),
            old_logprobs=((-0.4,),),
            reference_logprobs=((-0.4,),),
            advantages=(0.0,),
            response_mask=((True,),),
            beta=0.1,
        )
        self.assertAlmostEqual(terms.kl_loss, 0.0)
        self.assertAlmostEqual(terms.total_loss, 0.0)

    def test_rejects_empty_effective_batch(self):
        with self.assertRaisesRegex(ValueError, "at least one valid token"):
            grpo_loss_reference(
                new_logprobs=((-0.1,),),
                old_logprobs=((-0.1,),),
                advantages=(1.0,),
                response_mask=((False,),),
            )


if __name__ == "__main__":
    unittest.main()
