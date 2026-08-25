"""Runtime safety patch for Strategy 2 Megatron checkpoints.

Python imports ``sitecustomize`` during startup when this directory is on
``PYTHONPATH``.  The affected Megatron build implements even a synchronous
checkpoint save through ``FileSystemWriterAsync`` and asks it to stage GPU
tensors with a non-blocking device-to-host copy.  On this L20 host that path
intermittently raises ``CUDA error: invalid argument`` after training has
already completed, leaving only an incomplete checkpoint directory.

Set ``RETOOL_SYNC_CKPT_STAGING=1`` for training jobs to make that staging copy
blocking.  This changes checkpoint I/O latency only; it does not change model
weights, rewards, optimizer configuration, or rollout behavior.
"""

from __future__ import annotations

import os


def _force_blocking_checkpoint_staging() -> None:
    if os.environ.get("RETOOL_SYNC_CKPT_STAGING") != "1":
        return

    from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

    original_preload = FileSystemWriterAsync.preload_tensors

    def preload_tensors_blocking(write_buckets, non_blocking=True):
        # Keep the upstream signature because Megatron passes ``True`` here;
        # intentionally override only that unsafe transfer mode.
        return original_preload(write_buckets, non_blocking=False)

    FileSystemWriterAsync.preload_tensors = staticmethod(preload_tensors_blocking)


_force_blocking_checkpoint_staging()
