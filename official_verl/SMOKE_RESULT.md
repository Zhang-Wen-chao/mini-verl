# Official verl L20 GRPO smoke — 2026-08-18

## Outcome

The official `verl` FSDP2 + vLLM GRPO training loop ran through all **16/16**
optimizer steps on four NVIDIA L20 GPUs. It produced complete FSDP actor and
optimizer checkpoint shards through `global_step_16`. This confirms the target
systems path—not model quality—on the L20 host.

The historical launcher did not persist its final exit code. After the last
checkpoint, the Ray driver remained in cleanup while all GPU resources had
already been returned; the subsequent audit found no residual task processes.
The final checkpoint is intact and no OOM, CUDA error, or training traceback was
found. It is still classified as **systems-smoke complete / quality result
rejected**, not as a promotion candidate: this historical run lacks the new
launch-status evidence and its quality metrics are not acceptable. Future runs
write `logs/exit_status` from the launcher.

## Immutable inputs

- Official verl: `c4b389adadc58ce51cb2b63e70df497ca166d77f`
- Model: `Qwen/Qwen3-0.6B` revision
  `c1899de289a04d12100db370d81485cdf75e47ca`
- Model weight: 1,503,300,328 bytes; SHA-256
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`
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

## Measured evidence

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
- Ray cleanup retained the driver for several minutes after it had returned all
  GPU resources. The later audit found no residual task processes. Future runs
  record the launcher status so this phase is independently auditable.

## Next gate

Rerun the 0.6B smoke with persisted exit status and save raw samples. Then choose
a 4B instruct snapshot, recalculate the FSDP/vLLM memory budget, use a longer
response limit or a task-compatible format, and run a short 4B smoke before any
longer training. In parallel, map this exact official path into the single-node
`mini_verl` learning implementation; do not reproduce Ray/FSDP/vLLM orchestration
inside mini-verl.
