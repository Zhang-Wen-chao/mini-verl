import copy
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional locally")
class PolicySyncTest(unittest.TestCase):
    def test_synchronizes_independent_replicas_without_storage_aliasing(self):
        from mini_verl.policy_sync import synchronize_policy

        torch.manual_seed(17)
        source = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
        destination = copy.deepcopy(source)
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.add_(1.0)

        source_state = source.state_dict()
        destination_state = destination.state_dict()
        self.assertTrue(any(not torch.equal(source_state[key], destination_state[key]) for key in source_state))

        handle = synchronize_policy(source, destination, policy_version=12)

        self.assertEqual(handle.version, 12)
        self.assertEqual(handle.parameter_tensors, len(source_state))
        self.assertGreater(handle.parameter_bytes, 0)
        for key, source_value in source.state_dict().items():
            destination_value = destination.state_dict()[key]
            self.assertTrue(torch.equal(source_value, destination_value))
            self.assertNotEqual(source_value.data_ptr(), destination_value.data_ptr())

        with torch.no_grad():
            next(source.parameters()).add_(1.0)
        self.assertFalse(torch.equal(next(source.parameters()), next(destination.parameters())))

    def test_rejects_invalid_policy_version(self):
        from mini_verl.policy_sync import synchronize_policy

        model = torch.nn.Linear(1, 1)
        with self.assertRaisesRegex(ValueError, "policy_version"):
            synchronize_policy(model, copy.deepcopy(model), policy_version=-1)


if __name__ == "__main__":
    unittest.main()
