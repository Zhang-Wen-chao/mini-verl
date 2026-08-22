"""Runtime-only patches for the pinned official one-step trainer.

This module is loaded automatically by Python when the calibration launcher
adds this directory to ``PYTHONPATH``.  It must *not* import ``verl`` or
``torch`` at interpreter startup: Ray creates reusable worker processes and
assigns their CUDA visibility before executing the task.  The patches are
therefore registered as import hooks and are applied only after verl imports
the corresponding module inside its own normal startup sequence.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import mmap
import os
import sys
import tempfile


_ONE_STEP_MODULE = "verl.experimental.one_step_off_policy.ray_trainer"
_WORKER_MODULE = "verl.single_controller.base.worker"
_VLLM_ROLLOUT_MODULE = "verl.workers.rollout.vllm_rollout.vllm_rollout"
_BUCKET_TRANSFER_MODULE = "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"
_MATH_REWARD_MODULE = "verl.utils.reward_score.math_reward"
_MMAP_TRANSFER_ENV = "MINI_VERL_FORCE_MMAP_WEIGHT_TRANSFER"
_MMAP_TRANSFER_DIR_ENV = "MINI_VERL_WEIGHT_TRANSFER_MMAP_DIR"
_MATH_REWARD_MODE_ENV = "MINI_VERL_MATH_REWARD_MODE"


def _patch_one_step_module(module) -> None:
    """Apply narrow compatibility fixes to the pinned one-step trainer."""
    trainer_cls = module.OneStepOffRayTrainer
    if getattr(trainer_cls, "_mini_verl_dump_lifecycle_patch", False):
        return

    original_dump_generations = trainer_cls._dump_generations
    original_fit = trainer_cls.fit

    @functools.wraps(original_dump_generations)
    def dump_generations_with_lazy_executor(self, *args, **kwargs):
        if not hasattr(self, "_dump_executor"):
            self._init_dump_executor()
        return original_dump_generations(self, *args, **kwargs)

    @functools.wraps(original_fit)
    async def fit_with_dump_executor_cleanup(self, *args, **kwargs):
        try:
            return await original_fit(self, *args, **kwargs)
        finally:
            if hasattr(self, "_dump_executor"):
                self._shutdown_dump_executor()

    trainer_cls._dump_generations = dump_generations_with_lazy_executor
    trainer_cls.fit = fit_with_dump_executor_cleanup

    # The pinned one-step trainer creates its Critic as a standalone
    # TrainingWorker, whose registered initialization RPC is ``reset()``.
    # Calling ``init_model()`` only works for the actor/ref composite worker
    # groups, so the upstream one-step override fails before any PPO update.
    # Keep its actor/ref/reward setup intact and align only the Critic branch
    # with SeparateRayPPOTrainer._init_models().
    def init_models_with_standalone_critic(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(module.Role.Critic)]
            self.critic_wg.reset()
            from functools import partial

            from verl.workers.utils.losses import value_loss

            self.critic_wg.set_loss_fn(partial(value_loss, config=self.orig_critic_cfg))

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(module.Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = self.all_wg[str(module.Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_wg = self.all_wg[str(module.Role.Actor)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg

    trainer_cls._init_models = init_models_with_standalone_critic
    trainer_cls._mini_verl_dump_lifecycle_patch = True


def apply_one_step_generation_dump_patch() -> None:
    """Public helper retained for focused unit tests and interactive checks."""
    from verl.experimental.one_step_off_policy import ray_trainer

    _patch_one_step_module(ray_trainer)


def _patch_worker_module(module) -> None:
    """Log the physical device selected by Torch after Ray assignment."""
    if os.environ.get("MINI_VERL_DEBUG_GPU_MAPPING") != "1":
        return

    worker_cls = module.Worker
    if getattr(worker_cls, "_mini_verl_gpu_mapping_debug_patch", False):
        return

    original_setup = worker_cls._setup_env_cuda_visible_devices

    @functools.wraps(original_setup)
    def setup_with_gpu_mapping_evidence(self, *args, **kwargs):
        import ray

        def evidence(stage):
            try:
                accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
            except Exception as exc:  # pragma: no cover - defensive diagnostics
                accelerator_ids = {"error": repr(exc)}
            torch_bus = "unavailable"
            try:
                import torch

                torch_bus = torch.cuda.get_device_properties(torch.cuda.current_device()).pci_bus_id
            except Exception as exc:  # pragma: no cover - diagnostic only
                torch_bus = f"error:{exc!r}"
            print(
                "MINI_VERL_GPU_MAPPING "
                f"stage={stage} pid={os.getpid()} "
                f"rank={os.environ.get('RANK')} "
                f"local_rank={os.environ.get('LOCAL_RANK')} "
                f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"torch_bus={torch_bus} accelerator_ids={accelerator_ids}",
                flush=True,
            )

        result = original_setup(self, *args, **kwargs)
        evidence("after_worker_setup")
        return result

    worker_cls._setup_env_cuda_visible_devices = setup_with_gpu_mapping_evidence
    worker_cls._mini_verl_gpu_mapping_debug_patch = True


def apply_ray_gpu_mapping_debug_patch() -> None:
    """Public helper retained for focused unit tests and interactive checks."""
    if os.environ.get("MINI_VERL_DEBUG_GPU_MAPPING") != "1":
        return
    from verl.single_controller.base import worker

    _patch_worker_module(worker)


class _MMapSegment:
    """Small ``SharedMemory``-compatible wrapper backed by a regular file.

    CUDA IPC can be disallowed by a container's seccomp policy even though CUDA
    reports that the physical GPU supports it.  POSIX shared memory is not a
    viable fallback for Qwen3.5-4B either: its largest FP32 transfer tensor is
    larger than this container's 1 GiB ``/dev/shm`` mount.  A short-lived mmap
    file has the same buffer interface while using the container's normal
    filesystem instead.
    """

    def __init__(self, path: str, size: int, create: bool) -> None:
        self.path = path
        self.size = size
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        self._fd = os.open(path, flags, 0o600)
        try:
            if create:
                os.ftruncate(self._fd, size)
            actual_size = os.fstat(self._fd).st_size
            if actual_size < size:
                raise RuntimeError(f"mmap transfer file {path!r} is {actual_size} bytes, need {size}")
            self._mmap = mmap.mmap(self._fd, size, access=mmap.ACCESS_WRITE)
        except Exception:
            os.close(self._fd)
            raise
        self.buf = memoryview(self._mmap)

    def close(self) -> None:
        try:
            self.buf.release()
        except (AttributeError, BufferError):
            pass
        try:
            self._mmap.close()
        except BufferError:
            # Torch may still hold a temporary view while the worker finishes
            # its final device copy.  The file is unlinked below and the kernel
            # releases the mapping when that process drops the last view.
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass

    def unlink(self) -> None:
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


def _mmap_transfer_path(name: str) -> str:
    root = os.environ.get(_MMAP_TRANSFER_DIR_ENV, tempfile.gettempdir())
    return os.path.join(root, f"{name}.mmap")


def _patch_bucket_transfer_module(module) -> None:
    """Replace the official shared-memory primitive with mmap when opted in.

    The official bucket protocol, ZMQ acknowledgements and tensor loading stay
    unchanged.  Only the byte buffer backing its documented shared-memory
    fallback changes, so this is narrowly scoped to the container limitation.
    """
    if os.environ.get(_MMAP_TRANSFER_ENV) != "1":
        return
    if getattr(module, "_mini_verl_mmap_transfer_patch", False):
        return

    def create_mmap_transfer(size: int, name: str):
        os.makedirs(os.path.dirname(_mmap_transfer_path(name)), exist_ok=True)
        return _MMapSegment(_mmap_transfer_path(name), size=size, create=True)

    def rebuild_mmap_transfer(name: str, size: int, dtype=None):
        segment = _MMapSegment(_mmap_transfer_path(name), size=size, create=False)
        torch_dtype = module.torch.uint8 if dtype is None else dtype
        return module.torch.frombuffer(segment.buf[:size], dtype=torch_dtype), segment

    module.create_shared_memory = create_mmap_transfer
    module.rebuild_shared_memory = rebuild_mmap_transfer
    module._mini_verl_mmap_transfer_patch = True


def _patch_vllm_rollout_module(module) -> None:
    """Force the official vLLM adapter onto its non-CUDA-IPC path."""
    if os.environ.get(_MMAP_TRANSFER_ENV) != "1":
        return
    if getattr(module, "_mini_verl_force_mmap_transfer_patch", False):
        return
    module.is_support_ipc = lambda: False
    module._mini_verl_force_mmap_transfer_patch = True


def _patch_math_reward_module(module) -> None:
    """Opt in to exact rational equivalence without modifying pinned VeRL."""
    if os.environ.get(_MATH_REWARD_MODE_ENV) != "normalized":
        return
    if getattr(module, "_mini_verl_normalized_math_reward_patch", False):
        return
    from normalized_math_reward import compute_score as normalized_compute_score

    legacy_compute_score = module.compute_score

    @functools.wraps(legacy_compute_score)
    def compute_score_with_exact_rationals(solution_str, ground_truth):
        return normalized_compute_score(solution_str, ground_truth, legacy_compute_score)

    module.compute_score = compute_score_with_exact_rationals
    module._mini_verl_normalized_math_reward_patch = True


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader, patch):
        self._wrapped_loader = wrapped_loader
        self._patch = patch

    def create_module(self, spec):
        create_module = getattr(self._wrapped_loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module):
        self._wrapped_loader.exec_module(module)
        self._patch(module)


class _VerlPatchFinder(importlib.abc.MetaPathFinder):
    _patches = {
        _ONE_STEP_MODULE: _patch_one_step_module,
        _WORKER_MODULE: _patch_worker_module,
        _BUCKET_TRANSFER_MODULE: _patch_bucket_transfer_module,
        _VLLM_ROLLOUT_MODULE: _patch_vllm_rollout_module,
        _MATH_REWARD_MODULE: _patch_math_reward_module,
    }

    def find_spec(self, fullname, path=None, target=None):
        patch = self._patches.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _PatchLoader(spec.loader, patch)
        return spec


def install_delayed_patches() -> None:
    """Install hooks without importing verl/torch during Python site startup."""
    loaded_one_step = sys.modules.get(_ONE_STEP_MODULE)
    if loaded_one_step is not None:
        _patch_one_step_module(loaded_one_step)
    loaded_worker = sys.modules.get(_WORKER_MODULE)
    if loaded_worker is not None:
        _patch_worker_module(loaded_worker)
    loaded_bucket_transfer = sys.modules.get(_BUCKET_TRANSFER_MODULE)
    if loaded_bucket_transfer is not None:
        _patch_bucket_transfer_module(loaded_bucket_transfer)
    loaded_vllm_rollout = sys.modules.get(_VLLM_ROLLOUT_MODULE)
    if loaded_vllm_rollout is not None:
        _patch_vllm_rollout_module(loaded_vllm_rollout)
    loaded_math_reward = sys.modules.get(_MATH_REWARD_MODULE)
    if loaded_math_reward is not None:
        _patch_math_reward_module(loaded_math_reward)
    if not any(isinstance(finder, _VerlPatchFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _VerlPatchFinder())


install_delayed_patches()
