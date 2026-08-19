# mini-verl Runbook

## Local, no ML dependencies

The protocol, reward, GRPO reference implementation, controller and timing tests use only the Python standard library.

```bash
python3 -m unittest discover -s tests -v
PYTHONPATH=. python3 examples/phase0_smoke.py
```

PyTorch and Transformers tests are skipped rather than failed when optional dependencies are absent.

## Smallest complete GRPO run

Install only PyTorch, then run the package entry point. It downloads no model and
works on CPU; its categorical policy exists solely to make the complete GRPO
data flow inspectable.

```bash
python -m pip install -e '.[torch]'
python -m mini_verl.toy
# equivalent compatibility entry point
PYTHONPATH=. python examples/toy_grpo_train.py
```

The run must improve `final_pass@1` over `initial_pass@1`. Each iteration is
the same contract used by larger backends:

```text
sample G responses with old logprobs
→ rule reward
→ group-relative advantage
→ clipped GRPO update
→ policy version + 1
```

For a reproducible run, create a RunConfig with the seed, selected device and deterministic flag, then call seed_everything before model construction or rollout.

## GPU validation

Run the complete suite in an environment with PyTorch and Transformers.

```bash
CUDA_VISIBLE_DEVICES=0 python -m unittest discover -s tests -v
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python examples/toy_grpo_train.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python benchmarks/toy_grpo_benchmark.py --device cuda
```

The toy workload should improve greedy pass@1 from 0.125 to 1.000. This validates rollout, reward, group advantage and GRPO update plumbing; it does not measure language-model quality.

For a stage breakdown that exercises the actual Hugging Face `generate`, old-logprob recomputation and CausalLM GRPO update path without downloading a model, run the locally constructed tiny GPT-2 benchmark:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python benchmarks/tiny_hf_grpo_benchmark.py --device cuda \
  --iterations 20 --warmup 1 --repeats 3 \
  --sample-gpu-utilization --gpu-index 0
```

It uses fixed-seed sampling (rather than greedy decoding) because GRPO requires multiple responses per prompt. It reports per-iteration median rollout/reward/train time, response-token count, an opt-in forward-hook split of each `generate` call's first prefill forward versus subsequent cache decode forwards, and the separate full-sequence old-policy-logprob forward. The hook synchronizes CUDA before/after each measured forward, so use this diagnostic benchmark for attribution rather than maximum throughput. The 2-layer, 64-hidden-size model is deliberately too small for production conclusions; use it for integration and regression comparisons only.

To compare the correctness-first prompt micro-batch against the legacy per-prompt schedule, change only `--rollout-batch-size` (the worker left-pads variable-length prompts internally):

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python benchmarks/tiny_hf_grpo_benchmark.py --device cuda \
  --prompt-count 4 --group-size 2 --max-new-tokens 8 \
  --rollout-batch-size 1 --iterations 20 --warmup 1 --repeats 3

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python benchmarks/tiny_hf_grpo_benchmark.py --device cuda \
  --prompt-count 4 --group-size 2 --max-new-tokens 8 \
  --rollout-batch-size 4 --iterations 20 --warmup 1 --repeats 3
```

This is static prompt micro-batching, not a continuous-batching serving engine. Preserve the returned trajectory's original prompt ids and policy/version metadata when implementing a future vLLM/SGLang backend.

For variable-length prompts, the worker can additionally bucket prompt rows by
encoded length before forming static micro-batches.  It restores the output to
original prompt/sample order, so group ownership and downstream reward code do
not observe the scheduling change.  Compare it with the same synthetic prompt
distribution before enabling it in an experiment:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python benchmarks/hf_rollout_prompt_bucketing_benchmark.py --device cuda \
  --iterations 20 --warmup 1 --repeats 3 \
  --prompt-lengths 3,24,4,23,5,22,6,21 \
  --group-size 2 --max-new-tokens 8 --rollout-batch-size 4

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python benchmarks/hf_rollout_prompt_bucketing_benchmark.py --device cuda \
  --iterations 20 --warmup 1 --repeats 3 \
  --prompt-lengths 3,24,4,23,5,22,6,21 \
  --group-size 2 --max-new-tokens 8 --rollout-batch-size 4 \
  --bucket-prompts-by-length
```

Interpret `prompt_padding_ratio` together with wall time and real response-token
throughput.  Lower prompt padding does not guarantee a proportional latency win:
on a tiny model, Python/Transformers generation control flow and decode can
remain the dominant fixed cost.

`rollout_max_padded_prompt_tokens` adds a hard prefill-capacity guard on top of
the count limit.  Its unit is `group_size * prompt_rows * max_prompt_length`,
after `num_return_sequences` expansion, rather than raw request tokens.  A row
which cannot fit on its own is rejected before generation.  The following uses
an eight-row count cap but requires every generated micro-batch to stay within
180 expanded prompt tokens:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python benchmarks/hf_rollout_prompt_bucketing_benchmark.py --device cuda \
  --iterations 20 --warmup 1 --repeats 3 \
  --prompt-lengths 3,24,4,23,5,22,6,21 \
  --group-size 2 --max-new-tokens 8 --rollout-batch-size 8 \
  --bucket-prompts-by-length --rollout-max-padded-prompt-tokens 180
```

Check `max_batch_padded_prompt_tokens_median`, not only the cumulative padded
token count: it is the observable per-generate prefill peak controlled by this
guard.  This is not a complete KV-cache admission controller; the budget does
not include generated tokens, model activations, or decode-time cache growth.

When a fixed `max_new_tokens` reservation is required, use
`rollout_max_padded_sequence_tokens` instead (or together with the prompt
budget).  Its unit is `group_size * prompt_rows *
(max_prompt_length + max_new_tokens)`, so it bounds the scheduler's worst-case
sequence-token width per `generate` call.  It still is not a byte-accurate KV
cache estimator: model layout, dtype, attention implementation and temporary
activations are outside its scope.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python benchmarks/hf_rollout_prompt_bucketing_benchmark.py --device cuda \
  --iterations 20 --warmup 1 --repeats 3 \
  --prompt-lengths 3,24,4,23,5,22,6,21 \
  --group-size 2 --max-new-tokens 8 --rollout-batch-size 8 \
  --bucket-prompts-by-length --rollout-max-padded-sequence-tokens 240
```

Inspect `max_batch_padded_sequence_tokens_median`; a request that cannot fit
even alone is rejected before `model.generate`.

Before scheduling, the HF worker also checks `prompt_tokens + max_new_tokens`
against an exposed model context limit (`max_position_embeddings` or GPT-2's
`n_positions`).  This turns a fixed-length request that cannot possibly fit
into an explicit configuration error rather than a backend-dependent generation
failure.  Models that expose neither field are left to their backend; models
with dynamic RoPE scaling still need their deployment-specific context policy.

The same options are available on the end-to-end GRPO benchmark, so admission
cost is measured together with old-logprob recomputation, reward and training
rather than only in a rollout-only script:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python benchmarks/tiny_hf_grpo_benchmark.py --device cuda \
  --iterations 20 --warmup 1 --repeats 3 --prompt-count 8 \
  --prompt-lengths 3,24,4,23,5,22,6,21 \
  --group-size 2 --max-new-tokens 8 --rollout-batch-size 8 \
  --bucket-prompts-by-length --rollout-max-padded-sequence-tokens 240
```

Compare it against the identical command with the final budget option removed.
Report `iteration_seconds`, `rollout_seconds`, `train_seconds`, generation call
counts and both batch-peak token fields; a capacity guard is expected to reduce
batch consolidation and may reduce throughput.

## Trainer length micro-batching

`HuggingFaceTrainerWorker` also supports a training-side length-aware path:
`train_micro_batch_size` and `train_max_padded_tokens` sort a logical GRPO batch
by complete sequence length and split it into bounded micro-batches.  Gradients
are accumulated with each micro-loss weighted by its valid response-token share,
then a single optimizer step is performed.  This preserves the full logical
batch's token-mean GRPO objective; it is not gradient accumulation used to
silently turn one batch into multiple optimizer updates.

Use the end-to-end benchmark to compare memory/padding against its extra
forward/backward launches:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python benchmarks/tiny_hf_grpo_benchmark.py --device cuda \
  --iterations 20 --warmup 1 --repeats 3 --prompt-count 8 \
  --prompt-lengths 3,24,4,23,5,22,6,21 \
  --group-size 2 --max-new-tokens 8 --rollout-batch-size 8 \
  --bucket-prompts-by-length --rollout-max-padded-sequence-tokens 240 \
  --train-micro-batch-size 4 --train-max-padded-tokens 128
```

Compare `train_microbatch_count`, `train_real_sequence_tokens`,
`train_padded_sequence_tokens`, `train_padding_ratio`, `train_seconds`, and
allocator peaks.  The two-GPU pipeline benchmark accepts the same training
arguments; more trainer work can increase rollout overlap, but it still must be
judged by end-to-end wall time and resource limits.

The benchmark emits one JSON record containing configuration, median wall-clock throughput and (on CUDA) PyTorch allocator peak allocated/reserved bytes. Allocator peaks are not GPU utilization and must not be compared as such. To sample device-level utilization and used memory, first verify that the physical card is idle/exclusive, then opt in explicitly (the physical `nvidia-smi` index can differ from CUDA's logical index under `CUDA_VISIBLE_DEVICES`):

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python benchmarks/toy_grpo_benchmark.py --device cuda \
  --sample-gpu-utilization --gpu-index 0
```

The utilization fields are sampled every 0.1 s across all processes on that physical GPU; they are neither process-level nor a substitute for per-stage profiling. For a reproducible cited run, record the command, GPU model, visible-device setting, model/task, warmup/repeat count, and response-length distribution.

## Two-GPU trainer/rollout pipeline comparison

The following uses separate tiny GPT-2 replicas: learner on GPU 0, rollout on GPU 1. It compares a synchronous `rollout -> train -> full weight sync` schedule with the safe one-step-lag prefetch schedule. Run it only after confirming both physical cards are idle.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 \
python benchmarks/tiny_hf_pipeline_benchmark.py --pipeline synchronous \
  --trainer-device cuda:0 --rollout-device cuda:1 \
  --iterations 20 --warmup 1 --repeats 3 --rollout-batch-size 1 --sample-gpu-utilization \
  --trainer-gpu-index 0 --rollout-gpu-index 1

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 \
python benchmarks/tiny_hf_pipeline_benchmark.py --pipeline prefetch \
  --trainer-device cuda:0 --rollout-device cuda:1 \
  --iterations 20 --warmup 1 --repeats 3 --rollout-batch-size 1 --sample-gpu-utilization \
  --trainer-gpu-index 0 --rollout-gpu-index 1
```

For prefetch, `next_rollout_wall_seconds` is total submit-to-completion rollout time; `rollout_wait_seconds` is the tail left after learner work; `prefetch_overlap_seconds` is their bounded overlap. Prime cost is excluded from measured steady-state iterations.

To test whether rollout batching and trainer/rollout separation compose, hold all arguments fixed and change `--rollout-batch-size` from `1` to the prompt count (for the default workload, `4`). This reduces the amount of serial rollout work, so it may also reduce the absolute overlap PD separation can hide.

The two-GPU benchmark accepts the same variable-length prompt and admission
arguments as the single-GPU benchmark.  The following compares synchronous and
one-step-lag schedules under an explicit worst-case sequence capacity; both JSON
records include the actual `prompt_batch_count`, padded-token totals and per-batch
prompt/sequence peaks from the completed rollout worker.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 \
python benchmarks/tiny_hf_pipeline_benchmark.py --pipeline synchronous \
  --trainer-device cuda:0 --rollout-device cuda:1 \
  --iterations 20 --warmup 1 --repeats 3 --prompt-count 8 \
  --prompt-lengths 3,24,4,23,5,22,6,21 \
  --group-size 2 --max-new-tokens 8 --rollout-batch-size 8 \
  --bucket-prompts-by-length --rollout-max-padded-sequence-tokens 240

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 \
python benchmarks/tiny_hf_pipeline_benchmark.py --pipeline prefetch \
  --trainer-device cuda:0 --rollout-device cuda:1 \
  --iterations 20 --warmup 1 --repeats 3 --prompt-count 8 \
  --prompt-lengths 3,24,4,23,5,22,6,21 \
  --group-size 2 --max-new-tokens 8 --rollout-batch-size 8 \
  --bucket-prompts-by-length --rollout-max-padded-sequence-tokens 240
```

For prefetch, compare `next_rollout_wall_seconds`, `prefetch_overlap_seconds`
and `rollout_wait_seconds` with the synchronous rollout.  Never infer overlap
from the post-train wait alone, and ensure the rollout worker uses an independent
model replica as this benchmark does.

## Length bucketing comparison

This compares two batch compositions over exactly the same 32 variable-length trajectories and the same 8 optimizer steps per epoch. `mixed` interleaves response lengths 2/4/8/16 in each batch; `bucketed` groups them by length.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python benchmarks/hf_length_bucketing_benchmark.py --strategy mixed \
  --device cuda --epochs 20 --warmup 1 --repeats 3 \
  --sample-gpu-utilization --gpu-index 0

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python benchmarks/hf_length_bucketing_benchmark.py --strategy bucketed \
  --device cuda --epochs 20 --warmup 1 --repeats 3 \
  --sample-gpu-utilization --gpu-index 0
```

Compare `padding_ratio_median`, real/padded sequence-token throughput, train wall time and allocator peak together. The work uses the actual CausalLM GRPO trainer but a deliberately tiny local model, so it is a batching regression test rather than a production sizing result.

## Two-GPU DDP GRPO smoke

Before using shared GPUs, inspect nvidia-smi and choose idle cards. Use an explicit localhost rendezvous inside the container to avoid container hostname resolution issues.

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. \
torchrun --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29591 \
  tests/ddp_grpo_smoke.py
```

The result must report a world size of two and synchronized replicas.

To verify the same DDP contract through the actual Hugging Face CausalLM trainer rather than the categorical toy policy, replace the entry point with tests/ddp_hf_grpo_smoke.py.

## Checkpoints

Use save_checkpoint to atomically write model, optimizer, policy version and RNG state. Checkpoints are trusted local artifacts because optimizer deserialization uses PyTorch pickle semantics; never load one from an untrusted source.

## Policy synchronization

synchronize_policy copies a full state dict from the trainer model to an independent rollout replica and returns a versioned PolicyHandle. It is the correctness-first baseline for rollout and trainer living in separate processes. A production implementation should replace the full copy with a distributed or sharded transport only after preserving the same version contract.

## Hugging Face backend (reference extension)

HuggingFaceRolloutWorker is a correctness-first backend: it generates G responses per prompt and recomputes old-policy token logprobs with a causal-LM forward pass. HuggingFaceTrainerWorker repacks prompt plus response, extracts response-token logprobs, and applies masked GRPO with optional reference-policy KL.

For one end-to-end update with any complete *local* Hugging Face causal-LM
snapshot, run the generic smoke. Its deterministic reward is only a plumbing
check; use a real verifier before interpreting a metric as model quality.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python examples/hf_grpo_smoke.py --model /path/to/local/model
```

## Async rollout policy lag

AsyncRolloutBuffer runs rollout requests in worker threads and attaches every buffered batch to the policy version it requested. Before training, call consume_next with the learner policy version. A zero max_policy_lag preserves on-policy training; permit a positive lag only when the selected algorithm and experiment explicitly support it. Use independent rollout model replicas for concurrent Hugging Face generation.

PrefetchingController is the concrete one-step-lag policy. It submits rollout(v_k) from an independent rollout replica before learner work, waits for that rollout to finish, then publishes v_(k+1) to the replica. The next update consumes rollout v_k under learner v_(k+1). It therefore requires max_policy_lag of at least one and must not be used for a strictly on-policy experiment. Never use the same model instance for the rollout worker and synchronizer: the controller deliberately waits for generation to end before overwriting the rollout replica.

For async metrics, `rollout_wall_seconds` is a batch's submit-to-completion wall time and is the denominator of `rollout_tokens_per_second`; `rollout_wait_seconds` is only the post-training tail that failed to overlap. `StageTimings.rollout_seconds` has the latter meaning in this controller.
