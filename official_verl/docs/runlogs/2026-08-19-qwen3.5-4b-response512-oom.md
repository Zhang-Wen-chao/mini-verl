# Qwen3.5-4B GRPO calibration: 512-token response memory boundary

## Outcome

This is a retained, failed resource-calibration attempt—not a successful 4B
quality run. It proved that the selected official verl path can load and execute
one real Qwen3.5-4B GRPO update on four L20 GPUs, then located the memory limit
of the 512-token response setting.

## Fixed inputs and topology

- Official verl commit `c4b389adadc58ce51cb2b63e70df497ca166d77f`; locked local
  runtime `/tmp/official-verl-local-fsdp-vllm/venv`.
- Model: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
  official runtime reported `Qwen3_5ForConditionalGeneration` with 4.54B parameters.
- Train: fixed 2,048-row audited OpenR1-Math calibration parquet. Validation:
  fixed 64-row MATH-lighteval subset. Reward: pinned official math reward using
  the last `\boxed{...}` answer.
- Four NVIDIA L20 GPUs (44.52 GiB visible each): FSDP2 trainer GPUs 0--1,
  vLLM tensor-parallel rollout GPUs 2--3. Group size 4, train batch 2,
  prompt/response max 512/512, vLLM context 1024, two requested steps.

## What completed

- Preflight passed; FSDP2 loaded the 4.54B model and vLLM TP=2 loaded the same
  snapshot. Initial validation and rollout/validation sample persistence worked.
- Step 1 ran real generation, reference log-probs, GRPO update, validation and
  checkpointing. `global_step_1` contains two 10,350,349,971-byte model shards
  and two 18,157,113,047-byte optimizer shards.
- Step 1 showed the reward contract is still degenerate: training reward and
  advantages were zero, held-out MATH accuracy was 0/64, and all responses hit
  the 512-token cap. This is diagnostic evidence, not a quality score.

## Failure boundary

The process exited with status `1` during step 2 `loss.backward()` on GPU 0:
PyTorch could not allocate another 1.19 GiB with only 41.75 MiB free. PyTorch
had 37.60 GiB allocated and 322.42 MiB reserved-but-unallocated. All GPU
processes were released after the failure. This is a sequence/activation memory
limit for this configuration, not a model-loading or rollout compatibility error.

The follow-up keeps every other setting fixed and reduces only the response cap
to 256 (context 768), writing to a new artifact directory.
