# Official verl: first GRPO run

This directory makes the first **official** verl experiment reproducible without
vendoring a second copy of upstream. It deliberately owns the stable experiment
contract (task, reward, evaluation gates, and exact upstream commit), while the
algorithm configuration remains in the selected upstream revision. That avoids
silently applying an obsolete config schema to a fast-moving framework.

## Experiment boundary

The first actual official-framework target is `GRPO + verifiable GSM8K math
reward` with upstream's `Qwen3-0.6B` one-step-off-policy recipe on 4 L20 GPUs
(2 FSDP2 trainer + 2 vLLM rollout GPUs). It is a systems smoke, not the final
quality experiment. Start with the smoke limits in `experiment.json` (256 train /
64 evaluation examples, group size 4), then promote to the 4B instruct model only
after every gate in that file is satisfied. 8B remains an optional scale-up.

The official GSM8K recipe asks for a final `#### <integer>` answer and applies its
rule reward against the ground truth. Do not begin with a learned reward model or
subjective judging; otherwise the first failure cannot be localized.

## Why a pinned upstream checkout

`verl` evolves its package entrypoints and Hydra configuration fields. We therefore
do not hard-code a potentially stale command here. On the actual Linux/CUDA host,
we first check out an explicitly chosen release tag or commit, discover that
revision's GRPO example/config, record the immutable commit hash, and only then
copy/adapt its documented launch command. The copied command and tracked lock file
belong in the run log.

## On the Linux CUDA host

Use a Python version supported by the selected official revision (the preflight
currently accepts 3.10--3.12), a clean environment, and one or more NVIDIA GPUs.
The selected interpreter must expose `torch`, `transformers`, `ray`, and `vllm`
for the 4-GPU smoke. The preflight always reports those imports;
`--require-runtime` turns a missing module into a hard failure, preventing Ray
from reserving GPUs only to fail during worker startup. The probe performs an
actual import in a short child process, so it also detects a broken virtualenv
interpreter link or incompatible native extension.
The current macOS development machine cannot run this stage.

```bash
cd /path/to/mini-verl
# Do not use a floating branch name. Choose an official release tag or full SHA.
export VERL_REVISION=<official-release-tag-or-40-character-commit>
bash official_verl/bootstrap_verl.sh /workspace/verl
python official_verl/preflight.py \
  --verl-dir /workspace/verl --require-cuda --write-lock
```

The last command writes `official_verl/verl.lock.json`, recording the exact
resolved commit and GRPO files found upstream. It is intentionally **not** ignored:
add it to the experiment commit once the smoke run is accepted. Copy the relevant
upstream GRPO example/config into your run artifact directory, then reduce only its data
limits, group size, sequence limits, and model path to the smoke values in
`experiment.json`. Keep the upstream entrypoint and field names unchanged.

When Hugging Face access is unavailable on the L20 host, download OpenAI's public
GSM8K source JSONL and convert it into the exact row schema expected by the
official recipe with `prepare_openai_gsm8k.py`; this mirrors the selected
revision's `examples/data_preprocess/gsm8k.py` output rather than inventing a
parallel prompt or reward format.

## L20 4-GPU smoke

The checked-in `run_qwen3_0_6b_4gpu_smoke.sh` is an intentionally small, explicit
adaptation of the selected upstream `grpo_0.6b_gsm8k_fsdp2_2_6.sh`: two GPUs are
given to FSDP2 training and two to a tensor-parallel vLLM rollout. It uses 16
trajectories per optimizer batch and 4 samples per prompt. It is the first real
official-verl runtime gate before the 4B experiment, not a quality benchmark.

Prepare the fixed 256/64 data split with the adapter (the OpenAI raw JSONL is
downloaded separately, because this L20 host cannot reach Hugging Face):

```bash
python official_verl/prepare_openai_gsm8k.py \
  --input .official-verl/data/raw-gsm8k/train.jsonl \
  --output .official-verl/data/gsm8k-smoke/train.parquet --split train --limit 256
python official_verl/prepare_openai_gsm8k.py \
  --input .official-verl/data/raw-gsm8k/test.jsonl \
  --output .official-verl/data/gsm8k-smoke/test.parquet --split test --limit 64

export VERL_DIR="$PWD/.official-verl/verl"
export MODEL_PATH=/absolute/path/to/Qwen3-0.6B
export TRAIN_FILE="$PWD/.official-verl/data/gsm8k-smoke/train.parquet"
export TEST_FILE="$PWD/.official-verl/data/gsm8k-smoke/test.parquet"
bash official_verl/run_qwen3_0_6b_4gpu_smoke.sh
```

Before the first full run, require the following evidence:

- Base-model held-out accuracy was recorded with the same prompt and verifier.
- A rollout batch gives a non-degenerate reward distribution (not all 0 or all 1).
- Training metrics contain reward, KL, response length, loss, and checkpoint path.
- The post-training held-out evaluation and several raw generated answers are saved.

Immediately before the actual launch, re-run the check in locked mode. This guards
against a checkout that was moved after the config was prepared:

```bash
python official_verl/preflight.py \
  --verl-dir /workspace/verl --require-cuda --require-runtime --require-lock
```

`run_qwen3_0_6b_4gpu_smoke.sh` repeats this locked preflight with the exact
`python` that will launch the job, before it calls the official entrypoint.
It also persists the launcher's numeric exit code in `logs/exit_status`; a
checkpoint alone is not sufficient evidence that Ray shut down cleanly.

## Observed L20 smoke

The first historical run completed all 16 planned optimizer steps with the pinned
official source (`c4b389adadc58ce51cb2b63e70df497ca166d77f`), Qwen3-0.6B, 256
training rows, 64 held-out rows, and the documented 2 FSDP2 + 2 vLLM topology.
`global_step_16` contains both FSDP model and optimizer shards and the final
driver log contains the step-16 GRPO metrics. The run is therefore valid proof
that the official data → rollout → rule reward → GRPO update → checkpoint path
executes on this L20 host.

It is deliberately **not** a quality result: the pre-training validation score
was 1/64 (1.5625%) and the final validation score was 0/64. The tiny data budget,
one epoch, and 256-token response cap are only a systems smoke. Ray's driver
exited after the last checkpoint with a cleanup-side worker connection error, so
there is no trustworthy launcher exit status for this first historical run. The
checkpointed training loop completed, but its launcher did not preserve a clean
exit status.

A follow-up run on 2026-08-19 used the updated launcher and a container-local
runtime whose upstream commit and `uv.lock` exactly matched the persistent pinned
source. It completed 16/16 steps, saved `global_step_16`, ran initial and final
held-out evaluations, and wrote `logs/exit_status = 0`. That is the accepted
systems proof for this branch. Both baseline and final held-out scores were 0/64,
so it remains a systems result rather than a learning-quality result. The full
provenance and exact metrics are in `SMOKE_RESULT.md`.
The full artifact record is checked in as
`RUNLOG_2026-08-18_qwen3_0.6b_gsm8k_grpo_4gpu_smoke.md` and copied beside the
remote checkpoints as `RUNLOG.md`; `RUNLOG_2026-08-18_L20.md` is its compact
index.

## Local-disk environment on the L20 host

The initial 4-GPU runs were executed from a container-local virtual environment
at `/tmp/official-verl-local-fsdp-vllm`. This is deliberate: unpacking CUDA
extensions through `uv` on the shared filesystem was very slow. The local
environment is rebuilt by `bootstrap_local_official_env.sh` from the same pinned
official source and `uv.lock`; the model, parquet data, checkpoints, and run logs
remain under persistent `.official-verl/` storage. It is therefore disposable,
not an untracked dependency.

The source lock includes optional Megatron-related packages such as `mbridge`.
They stay installed for lock fidelity, but the FSDP2 + vLLM smoke only imports
`torch`, `transformers`, `ray`, `vllm`, and `verl` during runtime preflight.
Importing `mbridge` would incorrectly require the optional Megatron backend.

## Hand-off to mini-verl

Once this smoke run works, trace only this exact upstream path:

```text
dataset → rollout → verifier reward → group advantage
        → actor/reference log-probabilities → GRPO update → metrics
```

Then reproduce that path in `mini_verl`; do not add Ray, multi-node scheduling,
or a generic reward-model service until the single task has a verified comparison.
