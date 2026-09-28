#!/usr/bin/env python3
"""Summarize one verl training log as a per-step table.

Column-oriented, matching the official console metrics schema: response length
and its cap ratio, the rule reward, the two gradient norms, entropy, KL loss,
allocated/reserved device memory and host memory. Validation rows land on their
own `step:0..N` line, so a run with trainer.val_before_train=True shows the
base-model score as step 0.

Usage:
    python official_verl/analyze_verl_quickstart_steps.py artifacts/<run>/logs/train.log
"""

import re
import sys

KEYS = [
    "response_length/mean",
    "response_length/clip_ratio",
    "critic/score/mean",
    "critic/score/max",
    "actor/grad_norm",
    "critic/grad_norm",
    "actor/entropy",
    "actor/kl_loss",
    "global_seqlen/mean",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "perf/cpu_memory_used_gb",
]

HEADER = (
    "step   len   clip   score  max   a_gn    c_gn    ent      kl   "
    "mem_alloc mem_rsv  cpu_gb   val_acc"
)

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse(path):
    rows = []
    with open(path, errors="ignore") as handle:
        for line in handle:
            line = ANSI.sub("", line)
            if not re.search(r"step:\d+ - ", line):
                continue

            row = {"step": int(re.search(r"step:(\d+) - ", line).group(1))}
            for key in KEYS:
                found = re.search(re.escape(key) + r":([-0-9.eE]+)", line)
                if found:
                    row[key] = float(found.group(1))

            # Reported as val-core/... by v0.7.1; older logs use val/....
            for pattern in (
                r"val-core/openai/gsm8k/acc/mean@1:([-0-9.eE]+)",
                r"val/test_score/openai/gsm8k:([-0-9.eE]+)",
            ):
                found = re.search(pattern, line)
                if found:
                    row["val_acc"] = float(found.group(1))
            rows.append(row)
    return rows


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__.strip().splitlines()[-1].strip())

    rows = parse(sys.argv[1])
    if not rows:
        sys.exit(f"no 'step:N - ' metric lines found in {sys.argv[1]}")

    print(HEADER)
    for row in rows:
        print(
            f"{row['step']:>4} "
            f"{row.get('response_length/mean', 0):>6.1f} "
            f"{row.get('response_length/clip_ratio', 0):>6.3f} "
            f"{row.get('critic/score/mean', 0):>6.4f} "
            f"{row.get('critic/score/max', 0):>4.1f} "
            f"{row.get('actor/grad_norm', 0):>6.2f} "
            f"{row.get('critic/grad_norm', 0):>7.1f} "
            f"{row.get('actor/entropy', 0):>5.3f} "
            f"{row.get('actor/kl_loss', 0):>6.4f} "
            f"{row.get('perf/max_memory_allocated_gb', 0):>9.2f} "
            f"{row.get('perf/max_memory_reserved_gb', 0):>7.2f} "
            f"{row.get('perf/cpu_memory_used_gb', 0):>7.1f} "
            f"{row.get('val_acc', '')}"
        )


if __name__ == "__main__":
    main()
