import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "official_verl" / "preflight.py"
SPEC = importlib.util.spec_from_file_location("official_verl_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class OfficialVerlPreflightTests(unittest.TestCase):
    def test_discovers_grpo_in_supported_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "config.yaml").write_text("algorithm: grpo\n", encoding="utf-8")
            (repo / "ignore.txt").write_text("grpo\n", encoding="utf-8")
            self.assertEqual(preflight.discover_grpo_files(repo), ["config.yaml"])

    def test_unversioned_directory_has_no_git_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(preflight.git_revision(Path(directory)))

    def test_codeload_snapshot_revision_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "UPSTREAM_COMMIT").write_text(("c" * 40) + "\n", encoding="utf-8")
            self.assertEqual(preflight.source_revision(repo), "c" * 40)

    def test_lock_must_match_the_resolved_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "verl.lock.json"
            lock.write_text('{"verl_revision": "' + ("a" * 40) + '"}', encoding="utf-8")
            report = {"verl_revision": "a" * 40}
            self.assertTrue(preflight.lock_matches(lock, report))
            self.assertFalse(preflight.lock_matches(lock, {"verl_revision": "b" * 40}))
            self.assertFalse(preflight.lock_matches(Path(directory) / "missing.json", report))

    def test_runtime_profile_reports_each_launch_dependency(self):
        profile = preflight.runtime_profile()
        self.assertEqual(set(profile["modules"]), set(preflight.RUNTIME_MODULES))
        self.assertIsInstance(profile["interpreter"], str)
        for module in profile["modules"].values():
            self.assertIsInstance(module["available"], bool)
            self.assertIn("version", module)
            self.assertIn("error", module)

    def test_runtime_availability_requires_every_dependency(self):
        available = {
            "modules": {name: {"available": True, "version": "test"} for name in preflight.RUNTIME_MODULES}
        }
        missing = {
            "modules": {name: {"available": True, "version": "test"} for name in preflight.RUNTIME_MODULES}
        }
        missing["modules"]["vllm"]["available"] = False
        self.assertTrue(preflight.runtime_available(available))
        self.assertFalse(preflight.runtime_available(missing))

    def test_runtime_is_reported_but_only_required_on_explicit_request(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "UPSTREAM_COMMIT").write_text(("c" * 40) + "\n", encoding="utf-8")
            (repo / "config.yaml").write_text("algorithm: grpo\n", encoding="utf-8")
            original = preflight.runtime_profile
            try:
                preflight.runtime_profile = lambda: {
                    "interpreter": "test",
                    "modules": {
                        name: {"available": False, "version": None}
                        for name in preflight.RUNTIME_MODULES
                    },
                }
                optional = preflight.build_report(repo, require_cuda=False)
                required = preflight.build_report(repo, require_cuda=False, require_runtime=True)
            finally:
                preflight.runtime_profile = original
        self.assertFalse(optional["checks"]["runtime_dependencies"])
        self.assertNotIn("runtime_dependencies", optional["hard_failures"])
        self.assertIn("runtime_dependencies", required["hard_failures"])

    def test_require_cuda_rejects_driver_incompatible_torch_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "UPSTREAM_COMMIT").write_text(("c" * 40) + "\n", encoding="utf-8")
            (repo / "config.yaml").write_text("algorithm: grpo\n", encoding="utf-8")
            original_cuda = preflight.cuda_profile
            try:
                preflight.cuda_profile = lambda: {
                    "nvidia_smi": True,
                    "query_ok": True,
                    "gpus": ["NVIDIA L20"],
                    "torch_cuda": {"is_available": False, "device_count": 0},
                }
                report = preflight.build_report(repo, require_cuda=True)
            finally:
                preflight.cuda_profile = original_cuda
        self.assertTrue(report["checks"]["cuda_visible"])
        self.assertFalse(report["checks"]["cuda_runtime_available"])
        self.assertIn("cuda_runtime_available", report["hard_failures"])


if __name__ == "__main__":
    unittest.main()
