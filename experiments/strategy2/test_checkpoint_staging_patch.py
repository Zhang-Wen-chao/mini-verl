"""Regression test for the L20 checkpoint-staging workaround."""

from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch


class CheckpointStagingPatchTests(unittest.TestCase):
    def test_enabled_patch_forces_blocking_device_to_host_copy(self):
        calls = []

        class Writer:
            @staticmethod
            def preload_tensors(write_buckets, non_blocking=True):
                calls.append((write_buckets, non_blocking))
                return "staged"

        module_names = [
            "megatron",
            "megatron.core",
            "megatron.core.dist_checkpointing",
            "megatron.core.dist_checkpointing.strategies",
            "megatron.core.dist_checkpointing.strategies.filesystem_async",
        ]
        modules = {name: types.ModuleType(name) for name in module_names}
        modules[module_names[-1]].FileSystemWriterAsync = Writer

        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"RETOOL_SYNC_CKPT_STAGING": "1"}, clear=False):
            sys.modules.pop("sitecustomize", None)
            patched = importlib.import_module("sitecustomize")
            self.assertEqual(Writer.preload_tensors(["bucket"], non_blocking=True), "staged")
            self.assertEqual(calls, [(["bucket"], False)])
            sys.modules.pop(patched.__name__, None)


if __name__ == "__main__":
    unittest.main()
