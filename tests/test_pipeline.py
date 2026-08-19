import unittest

from mini_verl.pipeline import AsyncRolloutBuffer
from mini_verl.protocol import Trajectory, TrajectoryBatch, TrajectoryValidationError


def batch(version: int) -> TrajectoryBatch:
    return TrajectoryBatch((
        Trajectory(
            prompt_token_ids=(1,),
            response_token_ids=(2,),
            old_logprobs=(-0.1,),
            policy_version=version,
            group_id="prompt",
        ),
    ))


class VersionedRolloutWorker:
    def __init__(self, *, returned_version: int | None = None):
        self.returned_version = returned_version
        self.calls = []

    def rollout(self, *, policy_version: int) -> TrajectoryBatch:
        self.calls.append(policy_version)
        return batch(policy_version if self.returned_version is None else self.returned_version)


class AsyncRolloutBufferTest(unittest.TestCase):
    def test_consumes_matching_on_policy_rollout(self):
        worker = VersionedRolloutWorker()
        with AsyncRolloutBuffer(worker) as buffer:
            buffer.submit(policy_version=3)
            result = buffer.consume_next(learner_policy_version=3)

        self.assertEqual(worker.calls, [3])
        self.assertEqual(result.requested_policy_version, 3)
        self.assertEqual(result.batch.policy_versions, frozenset({3}))
        self.assertGreaterEqual(result.rollout_wall_seconds, 0.0)

    def test_allows_explicitly_configured_one_step_policy_lag(self):
        with AsyncRolloutBuffer(VersionedRolloutWorker(), max_policy_lag=1) as buffer:
            buffer.submit(policy_version=3)
            result = buffer.consume_next(learner_policy_version=4)
        self.assertEqual(result.requested_policy_version, 3)

    def test_rejects_excessively_stale_or_misreported_rollouts(self):
        with AsyncRolloutBuffer(VersionedRolloutWorker(), max_policy_lag=0) as buffer:
            buffer.submit(policy_version=3)
            with self.assertRaisesRegex(TrajectoryValidationError, "lag"):
                buffer.consume_next(learner_policy_version=4)

        with AsyncRolloutBuffer(VersionedRolloutWorker(returned_version=4)) as buffer:
            buffer.submit(policy_version=3)
            with self.assertRaisesRegex(TrajectoryValidationError, "returned"):
                buffer.consume_next(learner_policy_version=3)

    def test_enforces_bounded_inflight_queue(self):
        with AsyncRolloutBuffer(VersionedRolloutWorker(), max_inflight=1) as buffer:
            buffer.submit(policy_version=0)
            with self.assertRaisesRegex(RuntimeError, "full"):
                buffer.submit(policy_version=0)


if __name__ == "__main__":
    unittest.main()
