import asyncio
import os
import runpy
import sys
import types
import unittest
from pathlib import Path


COMPAT = Path(__file__).resolve().parents[1] / "official_verl" / "compat" / "sitecustomize.py"


class OfficialVerlCompatTests(unittest.TestCase):
    def test_one_step_dump_patch_is_lazy_and_drained(self):
        class FakeTrainer:
            def _init_dump_executor(self):
                self._dump_executor = object()
                self.initialized = True

            def _shutdown_dump_executor(self):
                self.shutdown = True

            def _dump_generations(self, *args, **kwargs):
                self.dumped = True
                return "dumped"

            async def fit(self):
                self.fit_called = True
                return "fit"

        package_names = [
            "verl",
            "verl.experimental",
            "verl.experimental.one_step_off_policy",
            "verl.experimental.one_step_off_policy.ray_trainer",
        ]
        prior = {name: sys.modules.get(name) for name in package_names}
        try:
            ray_trainer = types.ModuleType(package_names[-1])
            ray_trainer.OneStepOffRayTrainer = FakeTrainer
            for name in package_names[:-1]:
                sys.modules[name] = types.ModuleType(name)
            sys.modules[package_names[-1]] = ray_trainer
            namespace = runpy.run_path(str(COMPAT), run_name="mini_verl_test_sitecustomize")
            namespace["apply_one_step_generation_dump_patch"]()
            trainer = FakeTrainer()
            self.assertEqual(trainer._dump_generations(), "dumped")
            self.assertTrue(trainer.initialized)
            self.assertEqual(asyncio.run(trainer.fit()), "fit")
            self.assertTrue(trainer.shutdown)
        finally:
            for name, module in prior.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_one_step_patch_uses_registered_reset_for_standalone_critic(self):
        source = COMPAT.read_text(encoding="utf-8")
        self.assertIn("def init_models_with_standalone_critic", source)
        self.assertIn("self.critic_wg.reset()", source)
        self.assertIn("self.critic_wg.set_loss_fn", source)
        self.assertNotIn("self.critic_wg.init_model()", source)

    def test_gpu_mapping_patch_is_opt_in(self):
        source = COMPAT.read_text(encoding="utf-8")
        self.assertIn('MINI_VERL_DEBUG_GPU_MAPPING', source)
        self.assertIn('MINI_VERL_GPU_MAPPING', source)
        old_value = os.environ.pop("MINI_VERL_DEBUG_GPU_MAPPING", None)
        try:
            namespace = runpy.run_path(str(COMPAT), run_name="mini_verl_test_sitecustomize_opt_out")
            # The guard must return before importing Ray/verl worker modules.
            self.assertIsNone(namespace["apply_ray_gpu_mapping_debug_patch"]())
        finally:
            if old_value is not None:
                os.environ["MINI_VERL_DEBUG_GPU_MAPPING"] = old_value

    def test_site_startup_registers_hooks_without_importing_verl(self):
        package_names = [name for name in sys.modules if name == "verl" or name.startswith("verl.")]
        prior = {name: sys.modules.pop(name) for name in package_names}
        try:
            namespace = runpy.run_path(str(COMPAT), run_name="mini_verl_test_sitecustomize_hooks")
            self.assertNotIn("verl", sys.modules)
            finder_type = namespace["_VerlPatchFinder"]
            self.assertTrue(any(isinstance(finder, finder_type) for finder in sys.meta_path))
        finally:
            sys.meta_path[:] = [
                finder
                for finder in sys.meta_path
                if finder.__class__.__module__ != "mini_verl_test_sitecustomize_hooks"
            ]
            for name, module in prior.items():
                sys.modules[name] = module

    def test_mmap_weight_transfer_patch_is_opt_in_and_does_not_import_torch(self):
        source = COMPAT.read_text(encoding="utf-8")
        self.assertIn("MINI_VERL_FORCE_MMAP_WEIGHT_TRANSFER", source)
        self.assertIn("_MMapSegment", source)
        namespace = runpy.run_path(str(COMPAT), run_name="mini_verl_test_sitecustomize_mmap")

        class FakeBucketModule:
            pass

        old_value = os.environ.pop("MINI_VERL_FORCE_MMAP_WEIGHT_TRANSFER", None)
        try:
            namespace["_patch_bucket_transfer_module"](FakeBucketModule)
            self.assertFalse(hasattr(FakeBucketModule, "_mini_verl_mmap_transfer_patch"))
        finally:
            if old_value is not None:
                os.environ["MINI_VERL_FORCE_MMAP_WEIGHT_TRANSFER"] = old_value


if __name__ == "__main__":
    unittest.main()
