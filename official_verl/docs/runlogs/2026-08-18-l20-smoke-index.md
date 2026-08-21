# Official verl GRPO run log — L20, 2026-08-18

This is the persistent record for the first official-verl L20 smoke. The exact
artifact copy is stored beside the remote checkpoints as `RUNLOG.md`.

## Identity

- `mini-verl` branch: `official-verl-grpo` (repository had no base commit)
- Official verl commit: `c4b389adadc58ce51cb2b63e70df497ca166d77f`
- Dependency lock: upstream `uv.lock` SHA-256
  `d353bb4ba73fb1089046734044860161050b469430455f4436df1cb916185470`
- Model: `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`
- Hardware: four NVIDIA L20 GPUs; FSDP2 on GPUs 0–1 and vLLM TP=2 on GPUs 2–3
- Runtime: Python 3.12, PyTorch 2.11.0+cu130, Transformers 5.5.3, Ray 2.55.1,
  vLLM 0.24.0, verl 0.10.0.dev

## Task and exact launch

- Dataset: OpenAI `grade-school-math`, converted into the selected official
  verl GSM8K Parquet schema by `prepare_openai_gsm8k.py`
- Immutable data: 256 train rows
  (`f1a6fabb7e0627226df614c9bb8363b0ee29dd0f4fcb468895b1b647994e9286`) and
  64 validation rows
  (`d43d0069f5d70fc6ac754ab5f013dc9eed6dec3f28f5efbbad609d07683b7892`)
- Algorithm: GRPO, official GSM8K integer rule reward, group size 4
- Limits: prompt/response 512/256 tokens; train batch 16; one epoch / 16 steps
- Actor: FSDP2, learning rate `1e-6`, reference KL coefficient `0.001`
- Rollout: vLLM, tensor parallel size 2, GPU memory utilization 0.60

The immutable launch definition is
`official_verl/run_qwen3_0_6b_4gpu_smoke.sh`, invoked as follows:

```bash
export VERL_DIR="$PWD/.official-verl/verl"
export MODEL_PATH="$PWD/.official-verl/models/Qwen3-0.6B"
export TRAIN_FILE="$PWD/.official-verl/data/gsm8k-smoke/train.parquet"
export TEST_FILE="$PWD/.official-verl/data/gsm8k-smoke/test.parquet"
bash official_verl/run_qwen3_0_6b_4gpu_smoke.sh
```

The effective entrypoint was
`python -m verl.experimental.one_step_off_policy.main_ppo`; exact resolved Hydra
arguments are present in `logs/train.log`.

## Observed results

- Initial deterministic validation: `1/64 = 0.015625` reward/accuracy.
- Every planned optimizer step completed and saved a checkpoint.
- Step 16 training reward: mean `0.015625`, min `0`, max `1`—the rollout reward
  signal was not degenerate.
- Step 16 actor PPO KL `0.0007782547`; KL loss `0.0020266087`; actor loss
  `0.0002129478`; gradient norm `0.5079324`.
- Step 16 mean response length `255.75`; response cap ratio `0.96875`.
- Step 16 duration `11.8229 s`; reported throughput `914.8356 tokens/s`.
- Final deterministic validation: `0/64 = 0.0` reward/accuracy.
- Last checkpoint: `checkpoints/global_step_16/`; two 1,503,446,571-byte model
  shards and two 2,384,226,344-byte optimizer shards.

## Decision and caveats

**Systems smoke accepted:** it proves the official GRPO/FSDP2/vLLM chain executes
end-to-end on this four-L20 topology.

**Quality result rejected:** a single 256-example epoch at a 256-token cap was
not designed to improve GSM8K, and it did not. This must not be represented as
model improvement or as a 4B result.

After the final checkpoint, the detached historical launcher did not retain an
exit code and Ray emitted a worker connection error while shutting down. The
driver subsequently exited and all four GPUs were released. No training-time OOM,
Python traceback, or incomplete final checkpoint was observed. The launch script
has since been changed to persist `logs/exit_status`; a clean 0.6B rerun with raw
samples and that status is the gate before a 4B smoke.

Platform warnings observed: optional Megatron backends unavailable, Gloo hostname
fallback, file-descriptor soft-limit advice, and L20 SM 8.9 SymmMem unavailable.
They did not prevent the completed 16-step loop.
