import copy
import time
import unittest

from mini_verl.async_controller import PrefetchingController
from mini_verl.pipeline import AsyncRolloutBuffer
from mini_verl.policy_sync import PolicyHandle
from mini_verl.protocol import Trajectory, TrajectoryBatch
from mini_verl.workers import RuleRewardWorker


def batch(version: int) -> TrajectoryBatch:
    return TrajectoryBatch((
        Trajectory(
            prompt_token_ids=(1,),
            response_token_ids=(2,),
            old_logprobs=(-0.1,),
            policy_version=version,
            group_id=f"prompt-{version}",
        ),
    ))


class RecordingRolloutWorker:
    def __init__(self):
        self.calls = []

    def rollout(self, *, policy_version: int) -> TrajectoryBatch:
        self.calls.append(policy_version)
        return batch(policy_version)


class RecordingTrainer:
    def __init__(self):
        self.versions = []

    def train(self, scored: TrajectoryBatch, *, learner_policy_version: int):
        self.versions.append((learner_policy_version, scored.policy_versions))
        return {"loss": 1.0}


class RecordingSynchronizer:
    def __init__(self):
        self.versions = []

    def synchronize(self, *, policy_version: int) -> PolicyHandle:
        self.versions.append(policy_version)
        return PolicyHandle(version=policy_version, parameter_tensors=1, parameter_bytes=4)


try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional locally")
class IndependentReplicaPrefetchingControllerTest(unittest.TestCase):
    def test_synchronizes_an_independent_rollout_replica_only_after_prefetch_completes(self):
        """A prefetch must see the old replica even if learner training finishes first."""
        from mini_verl.policy_sync import ModelPolicySynchronizer

        class ModelReadingRolloutWorker:
            def __init__(self, model):
                self.model = model
                self.observed_weights = []
                self.calls = []

            def rollout(self, *, policy_version: int) -> TrajectoryBatch:
                self.calls.append(policy_version)
                # The submitted prefetch overlaps the learner update.  If the
                # controller synchronizes too early, this later read observes
                # the new weight instead of the rollout policy's old weight.
                if len(self.calls) > 1:
                    time.sleep(0.01)
                self.observed_weights.append(float(self.model.weight.detach().item()))
                return batch(policy_version)

        class MutatingTrainer:
            def __init__(self, model):
                self.model = model
                self.calls = []

            def train(self, scored: TrajectoryBatch, *, learner_policy_version: int):
                self.calls.append((learner_policy_version, scored.policy_versions))
                with torch.no_grad():
                    self.model.weight.add_(1.0)
                return {"loss": 0.0}

        trainer_model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            trainer_model.weight.zero_()
        rollout_model = copy.deepcopy(trainer_model)
        rollout = ModelReadingRolloutWorker(rollout_model)
        trainer = MutatingTrainer(trainer_model)
        synchronizer = ModelPolicySynchronizer(trainer_model, rollout_model)

        with AsyncRolloutBuffer(rollout, max_policy_lag=1) as buffer:
            controller = PrefetchingController(
                buffer, RuleRewardWorker(lambda _: 1.0), trainer, synchronizer, policy_version=0
            )
            controller.run_iteration()
            controller.run_iteration()

        self.assertEqual(rollout.calls, [0, 0, 1])
        self.assertEqual(rollout.observed_weights, [0.0, 0.0, 1.0])
        self.assertEqual(trainer.calls, [(0, frozenset({0})), (1, frozenset({0}))])
        self.assertEqual(float(trainer_model.weight.detach().item()), 2.0)
        self.assertEqual(float(rollout_model.weight.detach().item()), 2.0)


class PrefetchingControllerTest(unittest.TestCase):
    def test_prefetches_v_k_while_training_and_consumes_it_at_one_step_lag(self):
        rollout = RecordingRolloutWorker()
        trainer = RecordingTrainer()
        synchronizer = RecordingSynchronizer()
        with AsyncRolloutBuffer(rollout, max_policy_lag=1) as buffer:
            controller = PrefetchingController(
                buffer, RuleRewardWorker(lambda _: 1.0), trainer, synchronizer, policy_version=0
            )
            first = controller.run_iteration()
            second = controller.run_iteration()

        self.assertEqual(rollout.calls, [0, 0, 1])
        self.assertEqual(trainer.versions, [(0, frozenset({0})), (1, frozenset({0}))])
        self.assertEqual(synchronizer.versions, [1, 2])
        self.assertEqual(first.policy_version, 0)
        self.assertEqual(first.next_policy_version, 1)
        self.assertEqual(second.policy_version, 1)
        self.assertEqual(second.next_policy_version, 2)
        self.assertEqual(second.metrics["rollout_policy_version"], 0.0)
        self.assertEqual(second.metrics["learner_policy_version"], 1.0)
        self.assertEqual(second.metrics["policy_lag"], 1.0)
        self.assertEqual(second.metrics["next_rollout_policy_lag"], 1.0)
        self.assertGreaterEqual(second.metrics["rollout_wall_seconds"], 0.0)
        self.assertGreaterEqual(second.metrics["rollout_wait_seconds"], 0.0)
        self.assertGreaterEqual(second.metrics["next_rollout_wall_seconds"], 0.0)
        self.assertGreaterEqual(second.metrics["prefetch_overlap_seconds"], 0.0)
        self.assertTrue(controller.is_primed)

    def test_rejects_on_policy_only_buffer(self):
        with AsyncRolloutBuffer(RecordingRolloutWorker(), max_policy_lag=0) as buffer:
            with self.assertRaisesRegex(ValueError, "max_policy_lag"):
                PrefetchingController(
                    buffer, RuleRewardWorker(lambda _: 1.0), RecordingTrainer(), RecordingSynchronizer()
                )


if __name__ == "__main__":
    unittest.main()
