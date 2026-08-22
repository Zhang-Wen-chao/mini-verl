"""Summarize a completed, paired no-std versus standard GRPO development run.

The fixed 64-row monitor is development-only.  This tool intentionally never
loads or scores the MATH held-out 200 set, so it cannot turn repeated algorithm
selection into a held-out claim.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"training/global_step:(\d+)(?:\D|$)")
NUMBER = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


@dataclass(frozen=True)
class RunSummary:
    label: str
    root: Path
    final_step: int
    rollout_count: int
    monitor_initial: float | None
    monitor_final: float | None
    finite_kl_steps: int
    mixed_groups: int
    mixed_nonzero_grad_groups: int
    mean_step_seconds: float | None


def read_clean(path: Path) -> str:
    return ANSI_ESCAPE.sub("", path.read_text(encoding="utf-8", errors="replace"))


def number_after(line: str, key: str) -> float | None:
    match = re.search(re.escape(key) + r":(?:np\.float64\()?" + NUMBER, line)
    return float(match.group(1)) if match else None


def monitor_at_step(text: str, step: int) -> float | None:
    # The upstream log also emits val-aux fields before val-core, and the final
    # validation shares its line with training/global_step.  Match the step
    # prefix plus the metric rather than assuming an adjacent field order.
    step_pattern = re.compile(rf"(?:^|\) )step:{step}(?:\D|$)")
    values = [
        number_after(line, "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1")
        for line in text.splitlines()
        if step_pattern.search(line) and "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1" in line
    ]
    values = [value for value in values if value is not None]
    return values[-1] if values else None


def summarize(label: str, root: Path, expected_step: int) -> RunSummary:
    log_path = root / "logs" / "train.log"
    if (root / "logs" / "exit_status").read_text().strip() != "0":
        raise ValueError(f"{label}: training exit_status is not 0")
    if (root / "logs" / "watch_exit_status").read_text().strip() != "0":
        raise ValueError(f"{label}: watcher exit status is not 0")
    if not (root / "checkpoints" / f"global_step_{expected_step}").is_dir():
        raise ValueError(f"{label}: missing global_step_{expected_step} checkpoint")

    text = read_clean(log_path)
    lines = [line for line in text.splitlines() if "training/global_step:" in line]
    steps = [int(match.group(1)) for line in lines if (match := STEP_RE.search(line))]
    if expected_step not in steps:
        raise ValueError(f"{label}: missing global step {expected_step} metric")

    finite_kl_steps = 0
    mixed_groups = 0
    mixed_nonzero_grad_groups = 0
    step_seconds: list[float] = []
    for line in lines:
        kl = number_after(line, "actor/ppo_kl")
        score = number_after(line, "critic/score/mean")
        grad = number_after(line, "actor/grad_norm")
        seconds = number_after(line, "timing_s/step")
        if kl is not None and kl == kl and abs(kl) != float("inf"):
            finite_kl_steps += 1
        if score is not None and grad is not None and 0.0 < score < 1.0:
            mixed_groups += 1
            mixed_nonzero_grad_groups += grad > 0.0
        if seconds is not None:
            step_seconds.append(seconds)

    rollout_count = len(list((root / "rollout_samples").glob("*.jsonl")))
    return RunSummary(
        label=label,
        root=root,
        final_step=expected_step,
        rollout_count=rollout_count,
        monitor_initial=monitor_at_step(text, 0),
        monitor_final=monitor_at_step(text, expected_step),
        finite_kl_steps=finite_kl_steps,
        mixed_groups=mixed_groups,
        mixed_nonzero_grad_groups=mixed_nonzero_grad_groups,
        mean_step_seconds=sum(step_seconds) / len(step_seconds) if step_seconds else None,
    )


def percentage(value: float | None) -> str:
    return "not logged" if value is None else f"{value:.4%}"


def seconds(value: float | None) -> str:
    return "not logged" if value is None else f"{value:.2f} s"


def render(no_std: RunSummary, standard: RunSummary) -> str:
    rows = []
    for run in (no_std, standard):
        rows.append(
            f"| {run.label} | {run.rollout_count} | {percentage(run.monitor_initial)} | "
            f"{percentage(run.monitor_final)} | {run.finite_kl_steps} | "
            f"{run.mixed_nonzero_grad_groups}/{run.mixed_groups} | {seconds(run.mean_step_seconds)} |"
        )
    return "\n".join(
        [
            "# Paired GRPO development-run summary",
            "",
            "## Scope",
            "",
            "This is a **single-seed development comparison**, not a held-out algorithm winner.",
            "Both runs use the same base model, 2,037 OpenR1-Math rows, 3 trainer + 1 rollout L20 topology, 4 samples per prompt, legacy reward, LR `1e-6`, reference KL `0.001`, and 170 steps. The sole intended algorithm variable is whether GRPO normalizes group advantages by their standard deviation.",
            "",
            "The monitor below is the fixed 64-row development set. The MATH held-out 200 set was not read or scored by this report.",
            "",
            "## Completion and telemetry",
            "",
            "| Run | rollout JSONL | monitor at step 0 | monitor at step 170 | finite-KL metric steps | mixed groups with non-zero gradient | mean logged step time |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## Interpretation guardrails",
            "",
            "- The monitor is useful for debugging and development selection only; repeated use makes it unsuitable as a final generalization estimate.",
            "- A later multi-seed confirmation must use a newly fixed development/final protocol before any MATH held-out comparison.",
            "- Completion, finite KL, and non-zero mixed-group gradients establish engineering health; they do not by themselves demonstrate superior learning quality.",
            "",
            "## Artifact roots",
            "",
            f"- no-std: `{no_std.root}`",
            f"- standard: `{standard.root}`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-std-root", type=Path, required=True)
    parser.add_argument("--standard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=170)
    args = parser.parse_args()

    no_std = summarize("no-std GRPO", args.no_std_root, args.steps)
    standard = summarize("standard GRPO", args.standard_root, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(no_std, standard), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
