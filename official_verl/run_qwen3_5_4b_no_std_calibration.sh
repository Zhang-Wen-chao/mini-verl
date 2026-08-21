#!/usr/bin/env bash
# Run the 3+1 Qwen3.5-4B GRPO calibration with only std normalization removed.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# The base launcher owns the model, memory and process-safety contract.  Keep
# the standard reward and all other defaults; this wrapper changes one upstream
# algorithm switch so the result remains an interpretable GRPO ablation.
exec bash "$SCRIPT_DIR/run_qwen3_5_4b_4gpu_calibration.sh" \
  algorithm.norm_adv_by_std_in_grpo=false \
  "$@"
