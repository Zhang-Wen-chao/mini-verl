import unittest

try:
    from benchmarks.hf_length_bucketing_benchmark import batches
except ModuleNotFoundError:
    batches = None


@unittest.skipIf(batches is None, "PyTorch and Transformers are optional locally")
class LengthBucketingBenchmarkTest(unittest.TestCase):
    def test_strategies_have_equal_work_but_different_padding(self):
        mixed = batches("mixed")
        bucketed = batches("bucketed")

        self.assertEqual(len(mixed), len(bucketed))
        self.assertEqual(
            sum(len(packed.batch.trajectories) for packed in mixed),
            sum(len(packed.batch.trajectories) for packed in bucketed),
        )
        self.assertEqual(
            sum(packed.real_sequence_tokens for packed in mixed),
            sum(packed.real_sequence_tokens for packed in bucketed),
        )
        self.assertGreater(
            sum(packed.padded_sequence_tokens for packed in mixed),
            sum(packed.padded_sequence_tokens for packed in bucketed),
        )
        self.assertAlmostEqual(
            sum(packed.padded_sequence_tokens for packed in bucketed)
            - sum(packed.real_sequence_tokens for packed in bucketed),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
