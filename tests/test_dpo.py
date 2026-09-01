import math
import unittest

from mini_verl.algorithms.dpo import DpoLossTerms, dpo_loss_reference


class DpoLossReferenceTest(unittest.TestCase):
    def test_preferred_chosen_response_produces_positive_margin_and_small_loss(self):
        terms = dpo_loss_reference(
            new_logprobs=((0.0, -1.0), (-2.0, -2.0)),
            reference_logprobs=((-2.0, 0.0), (-2.0, -1.0)),
            response_mask=((True, True), (True, True)),
            beta=1.0,
        )
        self.assertEqual(
            terms,
            DpoLossTerms(
                total_loss=0.12692801104297250,
                chosen_reward=1.0,
                rejected_reward=-1.0,
                reward_margin=2.0,
                accuracy=1.0,
                pair_count=1,
            ),
        )
        self.assertAlmostEqual(terms.total_loss, math.log1p(math.exp(-2.0)), places=15)

    def test_matching_reference_has_zero_margin_and_log_two_loss(self):
        terms = dpo_loss_reference(
            new_logprobs=((-1.0, -1.0), (-2.0, -2.0)),
            reference_logprobs=((-1.0, -1.0), (-2.0, -2.0)),
            response_mask=((True, True), (True, True)),
            beta=1.0,
        )
        self.assertEqual(
            terms,
            DpoLossTerms(
                total_loss=0.6931471805599453,
                chosen_reward=0.0,
                rejected_reward=0.0,
                reward_margin=0.0,
                accuracy=0.0,
                pair_count=1,
            ),
        )

    def test_mixed_margins_average_and_accuracy_count_both_directions(self):
        terms = dpo_loss_reference(
            new_logprobs=((1.0,), (0.0,), (0.0,), (1.0,)),
            reference_logprobs=((0.0,), (0.0,), (0.0,), (0.0,)),
            response_mask=((True,), (True,), (True,), (True,)),
            beta=1.0,
        )
        self.assertEqual(
            terms,
            DpoLossTerms(
                total_loss=0.8132616875182228,
                chosen_reward=0.5,
                rejected_reward=0.5,
                reward_margin=0.0,
                accuracy=0.5,
                pair_count=2,
            ),
        )

    def test_padding_tokens_do_not_change_any_loss_term(self):
        padded = dpo_loss_reference(
            new_logprobs=((0.0, -1.0, 5.0), (-2.0, -2.0, 5.0)),
            reference_logprobs=((-2.0, 0.0, 5.0), (-2.0, -1.0, 5.0)),
            response_mask=((True, True, False), (True, True, False)),
            beta=1.0,
        )
        unpadded = dpo_loss_reference(
            new_logprobs=((0.0, -1.0), (-2.0, -2.0)),
            reference_logprobs=((-2.0, 0.0), (-2.0, -1.0)),
            response_mask=((True, True), (True, True)),
            beta=1.0,
        )
        self.assertEqual(padded, unpadded)

    def test_length_normalize_divides_each_sequence_logprob_by_its_valid_tokens(self):
        inputs = dict(
            new_logprobs=((-1.0, -1.0), (-1.0, 7.0)),
            reference_logprobs=((0.0, 0.0), (0.0, 9.0)),
            response_mask=((True, True), (True, False)),
            beta=1.0,
        )
        unnormalized = dpo_loss_reference(**inputs)
        self.assertEqual(
            unnormalized,
            DpoLossTerms(
                total_loss=1.3132616875182228,
                chosen_reward=-2.0,
                rejected_reward=-1.0,
                reward_margin=-1.0,
                accuracy=0.0,
                pair_count=1,
            ),
        )
        normalized = dpo_loss_reference(**inputs, length_normalize=True)
        self.assertEqual(
            normalized,
            DpoLossTerms(
                total_loss=0.6931471805599453,
                chosen_reward=-1.0,
                rejected_reward=-1.0,
                reward_margin=0.0,
                accuracy=0.0,
                pair_count=1,
            ),
        )

    def test_rejects_non_positive_beta(self):
        for beta in (0.0, -0.1):
            with self.assertRaisesRegex(ValueError, "beta must be positive"):
                dpo_loss_reference(
                    new_logprobs=((0.0,), (0.0,)),
                    reference_logprobs=((0.0,), (0.0,)),
                    response_mask=((True,), (True,)),
                    beta=beta,
                )

    def test_rejects_odd_row_counts(self):
        with self.assertRaisesRegex(ValueError, "even number of rows"):
            dpo_loss_reference(
                new_logprobs=((0.0,), (0.0,), (0.0,)),
                reference_logprobs=((0.0,), (0.0,), (0.0,)),
                response_mask=((True,), (True,), (True,)),
            )

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "must match new_logprobs shape"):
            dpo_loss_reference(
                new_logprobs=((0.0, -1.0), (0.0, -1.0)),
                reference_logprobs=((0.0,), (0.0,)),
                response_mask=((True, True), (True, True)),
            )

    def test_rejects_non_finite_logprobs(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            dpo_loss_reference(
                new_logprobs=((0.0, math.inf), (0.0, 0.0)),
                reference_logprobs=((0.0, 0.0), (0.0, 0.0)),
                response_mask=((True, True), (True, True)),
            )

    def test_rejects_fully_masked_row(self):
        with self.assertRaisesRegex(ValueError, "at least one valid token in every row"):
            dpo_loss_reference(
                new_logprobs=((0.0, -1.0), (0.0, -1.0)),
                reference_logprobs=((0.0, 0.0), (0.0, 0.0)),
                response_mask=((False, False), (True, True)),
            )

    def test_rejects_non_boolean_mask(self):
        with self.assertRaisesRegex(ValueError, "only contain booleans"):
            dpo_loss_reference(
                new_logprobs=((0.0,), (0.0,)),
                reference_logprobs=((0.0,), (0.0,)),
                response_mask=((1,), (1,)),
            )


if __name__ == "__main__":
    unittest.main()
