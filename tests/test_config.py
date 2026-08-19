import random
import unittest

from mini_verl.config import RunConfig, seed_everything


class RunConfigTest(unittest.TestCase):
    def test_python_seed_is_reproducible_and_config_is_serializable(self):
        config = RunConfig(seed=21, device="cpu", deterministic=True)
        seed_everything(config)
        first = [random.random() for _ in range(4)]
        seed_everything(config)
        second = [random.random() for _ in range(4)]
        self.assertEqual(first, second)
        self.assertEqual(config.to_dict(), {"seed": 21, "device": "cpu", "deterministic": True})

    def test_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            RunConfig(seed=-1, device="cpu")
        with self.assertRaisesRegex(ValueError, "device"):
            RunConfig(seed=0, device="")


if __name__ == "__main__":
    unittest.main()
