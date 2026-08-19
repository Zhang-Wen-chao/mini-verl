import unittest

from mini_verl.metrics import StageTimer


class StageTimerTest(unittest.TestCase):
    def test_collects_expected_stages(self):
        timer = StageTimer()
        for stage in ("rollout", "reward", "train"):
            with timer.measure(stage):
                pass
        timings = timer.finish()
        self.assertEqual(set(timings.as_metrics()), {
            "rollout_seconds", "reward_seconds", "train_seconds", "sync_seconds", "iteration_seconds"
        })
        self.assertGreaterEqual(timings.iteration_seconds, 0.0)
        self.assertEqual(timings.sync_seconds, 0.0)

    def test_collects_optional_sync_stage(self):
        timer = StageTimer()
        for stage in ("rollout", "reward", "train", "sync"):
            with timer.measure(stage):
                pass
        self.assertGreaterEqual(timer.finish().sync_seconds, 0.0)

    def test_rejects_incomplete_or_duplicate_stages(self):
        timer = StageTimer()
        with timer.measure("rollout"):
            pass
        with self.assertRaisesRegex(ValueError, "already"):
            timer.measure("rollout")
        with self.assertRaisesRegex(ValueError, "missing"):
            timer.finish()

    def test_calls_optional_synchronizer_at_stage_boundaries(self):
        calls = []
        timer = StageTimer(synchronize=lambda: calls.append("sync"))
        for stage in ("rollout", "reward", "train"):
            with timer.measure(stage):
                pass
        timer.finish()
        self.assertEqual(calls, ["sync"] * 6)


if __name__ == "__main__":
    unittest.main()
