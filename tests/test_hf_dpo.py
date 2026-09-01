import copy
import unittest

from mini_verl.controller import Controller
from mini_verl.protocol import Trajectory, TrajectoryBatch
from mini_verl.workers import RuleRewardWorker

try:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch and Transformers are optional locally")
class HuggingFaceDpoTrainerTest(unittest.TestCase):
    def _model(self, *, vocab_size: int = 16, n_positions: int = 16):
        torch.manual_seed(3)
        return GPT2LMHeadModel(
            GPT2Config(
                vocab_size=vocab_size,
                n_positions=n_positions,
                n_embd=8,
                n_layer=1,
                n_head=1,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=1,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        )

    def _batch(self):
        return TrajectoryBatch((
            Trajectory((1, 2), (3, 4), (-2.0, -2.0), policy_version=2, group_id="a", reward=1.0),
            Trajectory((1, 2), (5,), (-2.0,), policy_version=2, group_id="a", reward=0.0),
            Trajectory((1, 2), (6, 7), (-2.0, -2.0), policy_version=2, group_id="b", reward=0.0),
            Trajectory((1, 2), (8,), (-2.0,), policy_version=2, group_id="b", reward=0.5),
        ))

    def test_dpo_trainer_updates_policy_and_keeps_reference_frozen(self):
        from mini_verl.hf import HuggingFaceDpoTrainerWorker

        model = self._model()
        reference = copy.deepcopy(model)
        policy_before = [parameter.detach().clone() for parameter in model.parameters()]
        reference_before = [parameter.detach().clone() for parameter in reference.parameters()]
        trainer = HuggingFaceDpoTrainerWorker(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
            pad_token_id=0,
            reference_model=reference,
        )
        metrics = trainer.train(self._batch(), learner_policy_version=2)

        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertEqual(metrics["pair_count"], 2.0)
        self.assertEqual(metrics["train_microbatch_count"], 1.0)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertTrue(
            any(
                not torch.equal(previous, current)
                for previous, current in zip(policy_before, model.parameters(), strict=True)
            )
        )
        for before_parameter, after_parameter in zip(reference_before, reference.parameters(), strict=True):
            self.assertTrue(torch.equal(before_parameter, after_parameter))

    def test_microbatched_pairs_match_full_batch_single_step(self):
        from mini_verl.hf import HuggingFaceDpoTrainerWorker

        full_model = self._model()
        micro_model = copy.deepcopy(full_model)
        full = HuggingFaceDpoTrainerWorker(
            model=full_model,
            optimizer=torch.optim.SGD(full_model.parameters(), lr=0.01),
            pad_token_id=0,
            reference_model=copy.deepcopy(full_model),
        )
        micro = HuggingFaceDpoTrainerWorker(
            model=micro_model,
            optimizer=torch.optim.SGD(micro_model.parameters(), lr=0.01),
            pad_token_id=0,
            reference_model=copy.deepcopy(full_model),
            train_micro_batch_size=1,
        )
        batch = self._batch()
        full_metrics = full.train(batch, learner_policy_version=2)
        micro_metrics = micro.train(batch, learner_policy_version=2)

        self.assertEqual(micro_metrics["train_microbatch_count"], 2.0)
        for key in ("loss", "chosen_reward", "rejected_reward", "reward_margin", "accuracy"):
            self.assertAlmostEqual(full_metrics[key], micro_metrics[key], places=6)
        for full_parameter, micro_parameter in zip(full_model.parameters(), micro_model.parameters(), strict=True):
            self.assertTrue(torch.allclose(full_parameter, micro_parameter, atol=2e-7, rtol=1e-5))

    def test_dpo_requires_a_reference_model(self):
        from mini_verl.hf import HuggingFaceDpoTrainerWorker

        model = self._model()
        with self.assertRaisesRegex(ValueError, "reference model"):
            HuggingFaceDpoTrainerWorker(
                model=model,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
                pad_token_id=0,
                reference_model=None,
            )

    def test_dpo_rejects_batch_without_valid_pairs(self):
        from mini_verl.hf import HuggingFaceDpoTrainerWorker

        model = self._model()
        trainer = HuggingFaceDpoTrainerWorker(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            pad_token_id=0,
            reference_model=copy.deepcopy(model),
        )
        tied = TrajectoryBatch((
            Trajectory((1, 2), (3, 4), (-2.0, -2.0), policy_version=2, group_id="a", reward=1.0),
            Trajectory((1, 2), (5,), (-2.0,), policy_version=2, group_id="a", reward=1.0),
        ))
        with self.assertRaisesRegex(ValueError, "at least one valid preference pair"):
            trainer.train(tied, learner_policy_version=2)

    def test_controller_runs_dpo_rollout_reward_and_update(self):
        from mini_verl.hf import HuggingFaceDpoTrainerWorker

        class FixedRolloutWorker:
            def __init__(self, batch: TrajectoryBatch):
                self.batch = batch

            def rollout(self, *, policy_version: int) -> TrajectoryBatch:
                return self.batch

        torch.manual_seed(5)
        model = self._model()
        batch = TrajectoryBatch((
            Trajectory((1, 2), (3, 4), (-2.0, -2.0), policy_version=0, group_id="a"),
            Trajectory((1, 2), (5,), (-2.0,), policy_version=0, group_id="a"),
        ))
        controller = Controller(
            rollout_worker=FixedRolloutWorker(batch),
            reward_worker=RuleRewardWorker(lambda trajectory: float(len(trajectory.response_token_ids) == 2)),
            trainer_worker=HuggingFaceDpoTrainerWorker(
                model=model,
                optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
                pad_token_id=0,
                reference_model=copy.deepcopy(model),
            ),
        )
        result = controller.run_iteration()

        self.assertEqual(result.policy_version, 0)
        self.assertEqual(result.next_policy_version, 1)
        self.assertEqual(result.trajectory_count, 2)
        self.assertAlmostEqual(result.mean_reward, 0.5)
        self.assertEqual(result.metrics["pair_count"], 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(result.metrics["loss"])))


if __name__ == "__main__":
    unittest.main()
