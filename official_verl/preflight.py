"""Validate a pinned official-verl GRPO experiment before spending GPU time.

This script intentionally does not assume a particular internal verl YAML schema.
That schema evolves with the upstream revision.  Instead it records the exact source
revision, discovers GRPO-related upstream files, and checks the stable experiment
contract owned by this repository.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "experiment.json"
LOCK_PATH = ROOT / "verl.lock.json"
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".rst", ".toml", ".sh"}
RUNTIME_MODULES = ("torch", "transformers", "ray", "vllm")


def run(command: list[str], cwd: Path | None = None) -> tuple[int, str]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return completed.returncode, completed.stdout.strip()


def git_revision(repo: Path) -> str | None:
    status, output = run(["git", "rev-parse", "HEAD"], cwd=repo)
    return output if status == 0 and re.fullmatch(r"[0-9a-f]{40}", output) else None


def source_revision(repo: Path) -> str | None:
    """Return a git commit, or the commit recorded for a codeload snapshot."""
    revision = git_revision(repo)
    if revision is not None:
        return revision
    try:
        recorded = (repo / "UPSTREAM_COMMIT").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return recorded if re.fullmatch(r"[0-9a-f]{40}", recorded) else None


def discover_grpo_files(repo: Path, limit: int = 20) -> list[str]:
    """Return bounded, relative source paths that mention GRPO."""
    matches: list[str] = []
    for path in repo.rglob("*"):
        if len(matches) >= limit:
            break
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "grpo" in content.lower() or "grpo" in path.name.lower():
            matches.append(str(path.relative_to(repo)))
    return matches


def cuda_profile() -> dict[str, Any]:
    profile: dict[str, Any] = {"nvidia_smi": shutil.which("nvidia-smi") is not None}
    if not profile["nvidia_smi"]:
        return profile
    status, output = run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    profile["query_ok"] = status == 0
    profile["gpus"] = output.splitlines() if status == 0 else []

    # Seeing a device in nvidia-smi is not sufficient to launch a distributed
    # PyTorch job.  The selected wheel can require a newer CUDA driver, in
    # which case Ray workers see zero CUDA devices and NCCL fails later.  Keep
    # this in a child process so a broken CUDA initialization cannot poison the
    # preflight process itself.  It deliberately constructs no model and
    # allocates no tensor.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'torch': torch.__version__, "
                "'torch_cuda': torch.version.cuda, "
                "'is_available': torch.cuda.is_available(), "
                "'device_count': torch.cuda.device_count()}))"
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0:
        try:
            profile["torch_cuda"] = json.loads(probe.stdout)
        except json.JSONDecodeError:
            profile["torch_cuda"] = {
                "is_available": False,
                "device_count": 0,
                "error": "CUDA probe returned invalid JSON",
            }
    else:
        lines = (probe.stderr or probe.stdout).strip().splitlines()
        profile["torch_cuda"] = {
            "is_available": False,
            "device_count": 0,
            "error": lines[-1] if lines else f"CUDA probe exited with status {probe.returncode}",
        }
    return profile


def runtime_profile() -> dict[str, Any]:
    """Report imports required by the pinned FSDP2 + vLLM smoke.

    Package metadata or a discoverable module does not guarantee that the selected
    interpreter can load it: stale environments may point to a missing interpreter
    or incompatible native extension. Probe each module in a short child process.
    This performs no model construction, Ray startup, or CUDA allocation.
    """
    modules: dict[str, dict[str, str | bool | None]] = {}
    for name in RUNTIME_MODULES:
        command = (
            "import importlib, importlib.metadata, json; "
            f"importlib.import_module({name!r}); "
            f"print(json.dumps({{\"version\": importlib.metadata.version({name!r})}}))"
        )
        probe = subprocess.run(
            [sys.executable, "-c", command],
            text=True,
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            try:
                version = json.loads(probe.stdout).get("version")
            except json.JSONDecodeError:
                version = None
            modules[name] = {"available": True, "version": version, "error": None}
        else:
            lines = (probe.stderr or probe.stdout).strip().splitlines()
            modules[name] = {
                "available": False,
                "version": None,
                "error": lines[-1] if lines else f"import exited with status {probe.returncode}",
            }
    return {"interpreter": sys.executable, "modules": modules}


def runtime_available(runtime: dict[str, Any]) -> bool:
    return all(bool(module["available"]) for module in runtime["modules"].values())


def build_report(
    verl_dir: Path,
    require_cuda: bool,
    require_runtime: bool = False,
) -> dict[str, Any]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    revision = source_revision(verl_dir) if verl_dir.is_dir() else None
    grpo_files = discover_grpo_files(verl_dir) if revision else []
    cuda = cuda_profile()
    runtime = runtime_profile()
    checks = {
        "linux": platform.system() == "Linux",
        "python_supported": (3, 10) <= sys.version_info[:2] <= (3, 12),
        "verl_directory": verl_dir.is_dir(),
        "verl_revision": revision is not None,
        "upstream_grpo_evidence": bool(grpo_files),
        "cuda_visible": bool(cuda.get("query_ok")),
        "cuda_runtime_available": bool(cuda.get("torch_cuda", {}).get("is_available"))
        and int(cuda.get("torch_cuda", {}).get("device_count", 0)) > 0,
        "runtime_dependencies": runtime_available(runtime),
    }
    hard_failures = [
        name
        for name in (
            "linux",
            "python_supported",
            "verl_directory",
            "verl_revision",
            "upstream_grpo_evidence",
        )
        if not checks[name]
    ]
    if require_cuda and not checks["cuda_visible"]:
        hard_failures.append("cuda_visible")
    if require_cuda and not checks["cuda_runtime_available"]:
        hard_failures.append("cuda_runtime_available")
    if require_runtime and not checks["runtime_dependencies"]:
        hard_failures.append("runtime_dependencies")
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "spec": spec["name"],
        "verl_dir": str(verl_dir.resolve()) if verl_dir.exists() else str(verl_dir),
        "verl_revision": revision,
        "source_kind": "git" if git_revision(verl_dir) else "codeload_snapshot" if revision else None,
        "grpo_candidates": grpo_files,
        "host": {"system": platform.system(), "machine": platform.machine(), "python": sys.version},
        "cuda": cuda,
        "runtime": runtime,
        "checks": checks,
        "hard_failures": hard_failures,
    }


def lock_matches(lock_path: Path, report: dict[str, Any]) -> bool:
    """Whether a previously recorded upstream revision matches this checkout."""
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return lock.get("verl_revision") == report.get("verl_revision")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-dir", type=Path, required=True, help="Pinned checkout of github.com/volcengine/verl")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless nvidia-smi and this PyTorch runtime can initialize CUDA",
    )
    parser.add_argument(
        "--require-runtime",
        action="store_true",
        help="Fail unless this Python can import torch, transformers, ray, and vllm",
    )
    parser.add_argument("--write-lock", action="store_true", help="Write the inspected upstream revision to verl.lock.json")
    parser.add_argument("--require-lock", action="store_true", help="Fail unless verl.lock.json matches this checkout")
    args = parser.parse_args()

    report = build_report(args.verl_dir, args.require_cuda, args.require_runtime)
    if args.require_lock and not lock_matches(LOCK_PATH, report):
        report["checks"]["matching_lock"] = False
        report["hard_failures"].append("matching_lock")
    elif args.require_lock:
        report["checks"]["matching_lock"] = True
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.write_lock and not report["hard_failures"]:
        LOCK_PATH.write_text(
            json.dumps(
                {
                    "verl_revision": report["verl_revision"],
                    "captured_at": report["generated_at"],
                    "grpo_candidates": report["grpo_candidates"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 1 if report["hard_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
