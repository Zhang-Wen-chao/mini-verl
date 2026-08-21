# Official verl GRPO clean rerun: Qwen3-0.6B-Base on four L20 GPUs

## Outcome

The updated official-verl launcher completed successfully: **16/16 optimizer
steps**, complete `global_step_16`, and `logs/exit_status = 0`. This is the clean
systems acceptance record for the official FSDP2 + vLLM + GRPO path. It is not a
quality result: initial and final held-out GSM8K accuracy@1 were both `0/64`.

## Identity

- Date: 2026-08-19 UTC.
- Official verl: `c4b389adadc58ce51cb2b63e70df497ca166d77f`.
- Lock SHA-256: `d353bb4ba73fb1089046734044860161050b469430455f4436df1cb916185470`.
- Runtime: `/tmp/official-verl-local-fsdp-vllm/venv`, rebuilt from that pinned
  source and lock. Runtime imports: PyTorch `2.11.0+cu130`, Transformers `5.5.3`,
  Ray `2.55.1`, vLLM `0.24.0`, verl `0.10.0.dev0`; CUDA 13.0, four GPUs visible.
- Model: `Qwen/Qwen3-0.6B-Base`; weights SHA-256
  `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`.
- Data: OpenAI grade-school-math adapted to official GSM8K rule-reward parquet:
  256 train, 64 validation; SHA-256 train
  `f1a6fabb7e0627226df614c9bb8363b0ee29dd0f4fcb468895b1b647994e9286`, validation
  `d43d0069f5d70fc6ac754ab5f013dc9eed6dec3f28f5efbbad609d07683b7892`.
- Hardware: 4 × NVIDIA L20 (46,068 MiB each), driver 550.127.05. GPUs 0--1:
  FSDP2 trainer. GPUs 2--3: vLLM TP=2 rollout.

## Configuration and result

- GRPO; group size 4; batch size 16; one epoch; max prompt/response 512/256;
  learning rate `1e-6`; low-variance KL coefficient `0.001`.
- Entrypoint: `python -m verl.experimental.one_step_off_policy.main_ppo`, via
  `official_verl/run_qwen3_0_6b_4gpu_smoke.sh`.
- Artifact:
  `/mnt/storage01/zhangwenchao02/repos/mini-verl-l20/artifacts/qwen3-0.6b-gsm8k-grpo-4gpu-smoke-rerun-localenv-20260819T1011`.
- Exit status: `0`; official total time: `515.29 s`.
- Initial held-out reward/accuracy@1: `0.0` (0/64). Final: `0.0` (0/64).
- Step 16: PPO KL `0.0011808624`; KL loss `0.0008903855`; actor loss
  `8.903855e-7`; mean response length `230.296875`; cap ratio `0.84375`; step
  time `17.4742 s`; throughput `572.3582 tokens/s`.
- `global_step_16` contains two model shards (1,503,446,571 bytes each), two
  optimizer shards (2,384,226,344 bytes each), rank state, tokenizer/config and
  `data.pt`; total size 7,786,812,460 bytes.

## Interpretation

No traceback, OOM, fatal error, or exception appeared in the training log, and
the launcher cleanly returned zero. The task still yielded zero rewards in the
observed smoke batches, while 84.375% of final-step responses hit the 256-token
limit. Preserve raw generated samples and establish a non-degenerate reward
distribution before treating a future 4B run as a quality experiment.
