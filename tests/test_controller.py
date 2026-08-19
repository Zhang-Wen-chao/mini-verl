import unittest

from mini_verl.controller import Controller
from mini_verl.protocol import Trajectory, TrajectoryBatch, TrajectoryValidationError
from mini_verl.workers import RuleRewardWorker


def trajectory(policy_version: int) -> Trajectory:
    return Trajectory(
        prompt_token_ids=(1,),
        response_token_ids=(2,),
        old_logprobs=(-0.1,),
        policy_version=policy_version,
        group_id="prompt",
    )


class StaticRolloutWorker:
    def __init__(self, *, returned_version: int):
        self.returned_version = returned_version
        self.calls: list[int] = []

    def rollout(self, *, policy_version: int) -> TrajectoryBatch:
        self.calls.append(policy_version)
        return TrajectoryBatch((trajectory(self.returned_version),))


class RecordingTrainer:
    def __init__(self):
        self.batches = []
        self.versions = []

    def train(self, batch: TrajectoryBatch, *, learner_policy_version: int):
        self.batches.append(batch)
        self.versions.append(learner_policy_version)
        return {"loss": 1.5}


class RecordingSynchronizer:
    def __init__(self, *, returned_version: int | None = None):
        self.returned_version = returned_version
        self.versions = []

    def synchronize(self, *, policy_version: int):
        from mini_verl.policy_sync import PolicyHandle

        self.versions.append(policy_version)
        return PolicyHandle(
            version=policy_version if self.returned_version is None else self.returned_version,
            parameter_tensors=2,
            parameter_bytes=64,
        )


class ControllerTest(unittest.TestCase):
    def test_runs_ordered_iteration_and_advances_policy_version(self):
        rollout = StaticRolloutWorker(returned_version=4)
        trainer = RecordingTrainer()
        controller = Controller(rollout, RuleRewardWorker(lambda _: 1.0), trainer, policy_version=4)

        result = controller.run_iteration()

        self.assertEqual(rollout.calls, [4])
        self.assertEqual(trainer.versions, [4])
        self.assertEqual(trainer.batches[0].trajectories[0].reward, 1.0)
        self.assertEqual(trainer.batches[0].trajectories[0].advantage, 0.0)
        self.assertEqual(result.policy_version, 4)
        self.assertEqual(result.next_policy_version, 5)
        self.assertEqual(controller.policy_version, 5)
        self.assertEqual(result.metrics["loss"], 1.5)
        self.assertEqual(result.metrics["mean_reward"], 1.0)
        self.assertEqual(result.metrics["mean_response_tokens"], 1.0)
        self.assertIn("rollout_tokens_per_second", result.metrics)
        self.assertIn("train_tokens_per_second", result.metrics)
        self.assertGreaterEqual(result.timings.rollout_seconds, 0.0)
        self.assertGreaterEqual(result.timings.reward_seconds, 0.0)
        self.assertGreaterEqual(result.timings.train_seconds, 0.0)
        self.assertGreaterEqual(result.timings.iteration_seconds, 0.0)
        self.assertEqual(result.policy_handle, None)

    def test_synchronizes_the_next_policy_version_after_training(self):
        rollout = StaticRolloutWorker(returned_version=4)
        trainer = RecordingTrainer()
        synchronizer = RecordingSynchronizer()
        controller = Controller(
            rollout,
            RuleRewardWorker(lambda _: 1.0),
            trainer,
            policy_version=4,
            policy_synchronizer=synchronizer,
        )

        result = controller.run_iteration()

        self.assertEqual(synchronizer.versions, [5])
        self.assertEqual(result.policy_handle.version, 5)
        self.assertEqual(result.policy_handle.parameter_bytes, 64)
        self.assertGreaterEqual(result.timings.sync_seconds, 0.0)
        self.assertIn("sync_seconds", result.metrics)

    def test_rejects_wrong_version_from_synchronizer_without_advancing(self):
        rollout = StaticRolloutWorker(returned_version=4)
        trainer = RecordingTrainer()
        synchronizer = RecordingSynchronizer(returned_version=7)
        controller = Controller(
            rollout,
            RuleRewardWorker(lambda _: 1.0),
            trainer,
            policy_version=4,
            policy_synchronizer=synchronizer,
        )

        with self.assertRaisesRegex(TrajectoryValidationError, "published version"):
            controller.run_iteration()
        self.assertEqual(controller.policy_version, 4)

    def test_rejects_stale_rollout_before_reward_or_training(self):
        rollout = StaticRolloutWorker(returned_version=3)
        trainer = RecordingTrainer()
        controller = Controller(rollout, RuleRewardWorker(lambda _: 1.0), trainer, policy_version=4)

        with self.assertRaisesRegex(TrajectoryValidationError, "incompatible"):
            controller.run_iteration()
        self.assertEqual(trainer.batches, [])
        self.assertEqual(controller.policy_version, 4)


if __name__ == "__main__":
    unittest.main()
