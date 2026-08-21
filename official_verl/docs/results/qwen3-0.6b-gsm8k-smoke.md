# Official verl L20 GRPO smoke

## Outcome

The official `verl` FSDP2 + vLLM GRPO training loop ran through all **16/16**
optimizer steps on four NVIDIA L20 GPUs. It produced complete FSDP actor and
optimizer checkpoint shards through `global_step_16`. This confirms the target
systems path—not model quality—on the L20 host.

The historical 2026-08-18 launcher did not persist its final exit code. Its final
checkpoint is intact, but it is retained as historical evidence only. A follow-up
rerun on 2026-08-19 used the updated launcher, completed all 16 steps, and wrote
`logs/exit_status = 0`. The clean rerun is the acceptance evidence for the
official training path. Both runs remain **systems-smoke complete / quality result
rejected**: their tiny data budget and zero held-out accuracy do not justify a
4B promotion.

## Clean rerun: 2026-08-19

- Artifact: `/mnt/storage01/zhangwenchao02/repos/mini-verl-l20/artifacts/qwen3-0.6b-gsm8k-grpo-4gpu-smoke-rerun-localenv-20260819T1011`.
- Launcher result: `logs/exit_status` contains `0`; total wall time reported by
  official verl was 515.29 seconds.
- Runtime was the container-local, rebuildable
  `/tmp/official-verl-local-fsdp-vllm/venv`. Its `UPSTREAM_COMMIT` and `uv.lock`
  SHA-256 matched the persistent official source before launch, and it imported
  PyTorch 2.11.0+cu130, Transformers 5.5.3, Ray 2.55.1, vLLM 0.24.0, and verl
  0.10.0.dev0 with four CUDA GPUs visible.
- Initial and final held-out GSM8K reward/accuracy@1 were both `0.0` (0/64).
- Step 16: PPO KL `0.0011808624`; KL loss `0.0008903855`; actor loss
  `8.903855e-7`; mean response length `230.296875`; response cap ratio `0.84375`;
  step time `17.4742 s`; throughput `572.3582 tokens/s`.
- `global_step_16` is complete: two model shards (1,503,446,571 bytes each), two
  optimizer shards (2,384,226,344 bytes each), rank state, tokenizer/config, and
  `data.pt` totaling 7,786,812,460 bytes.
- No traceback, OOM, fatal error, or exception was present in `train.log`.

This rerun proves the updated launcher, pinned runtime, FSDP2 trainer, TP=2 vLLM
rollout, validation, checkpointing, and clean launcher completion work together.
It does **not** demonstrate learning quality: rewards and group advantages were
zero on the observed smoke batches, held-out accuracy remained zero, and 84.375%
of final-step responses hit the 256-token cap. Raw generated samples were not
persisted by this run and are a required addition before a quality-oriented 4B
experiment.

## Immutable inputs

- Official verl: `c4b389adadc58ce51cb2b63e70df497ca166d77f`
- Clean-rerun model: `Qwen/Qwen3-0.6B-Base`; `model.safetensors` SHA-256
  `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`.
- The 2026-08-18 historical run used a separately recorded Qwen3-0.6B snapshot;
  see the [historical smoke runlog](../runlogs/2026-08-18-qwen3-0.6b-gsm8k-smoke.md)
  for that identity.
- Data: OpenAI `grade-school-math` source, converted with the selected
  upstream-compatible GSM8K adapter
- Train: 256 rows; SHA-256
  `f1a6fabb7e0627226df614c9bb8363b0ee29dd0f4fcb468895b1b647994e9286`
- Validation: 64 rows; SHA-256
  `d43d0069f5d70fc6ac754ab5f013dc9eed6dec3f28f5efbbad609d07683b7892`

## Configuration and topology

- Algorithm: GRPO with official GSM8K rule reward
- Four NVIDIA L20 GPUs: GPU 0–1 for FSDP2 actor/reference, GPU 2–3 for
  tensor-parallel vLLM rollout (`TP=2`)
- Group size 4; train batch size 16; one epoch; 16 total steps
- Maximum prompt/response lengths: 512 / 256 tokens
- Actor learning rate `1e-6`; reference KL loss coefficient `0.001`
- Runtime: container-local rebuildable environment at
  `/tmp/official-verl-local-fsdp-vllm`, sourced from the pinned checkout and
  the corresponding locked dependency graph (`uv.lock` SHA-256
  `d353bb4ba73fb1089046734044860161050b469430455f4436df1cb916185470`)

## Historical measured evidence: 2026-08-18

- Initial deterministic validation: reward/accuracy `0.015625` (1/64).
- Step 16 training reward mean: `0.015625` (non-degenerate; min `0`, max `1`).
- Step 16 actor PPO KL: `0.0007782547`; KL loss: `0.0020266087`; actor loss:
  `0.0002129478`; gradient norm: `0.5079324`.
- Step 16 mean response length: `255.75`; 96.875% hit the 256-token cap.
- Step 16 time: `11.8229 s`; reported throughput: `914.8356 tokens/s`.
- Final deterministic validation: reward/accuracy `0.0` (0/64).
- `global_step_16` shards:
  - `model_world_size_2_rank_{0,1}.pt`: 1,503,446,571 bytes each
  - `optim_world_size_2_rank_{0,1}.pt`: 2,384,226,344 bytes each

The short run was neither tuned nor long enough to assess quality. The high
response-cap ratio and zero final score are direct reasons to reject it as a
quality experiment.

## Known operational notes

- The successful runtime imports were PyTorch 2.11.0+cu130, Transformers 5.5.3,
  Ray 2.55.1, vLLM 0.24.0, and verl 0.10.0.dev.
- Benign platform warnings included unavailable optional Megatron backends, Gloo
  hostname fallback, file-descriptor soft-limit advice, and L20 SM 8.9
  SymmMem unavailability.
- The historical run retained a Ray driver for several minutes after it had
  returned all GPU resources. The clean rerun's launcher returned status zero.

## Next gate

Before considering 4B, add raw-sample persistence and an evaluation setup that
produces non-degenerate group rewards. Then choose a 4B instruct snapshot,
recalculate the FSDP/vLLM memory budget, use a task-compatible response limit,
and run a short 4B smoke. In parallel, map this exact official path into the
single-node `mini_verl` learning implementation; do not reproduce
Ray/FSDP/vLLM orchestration inside mini-verl.
