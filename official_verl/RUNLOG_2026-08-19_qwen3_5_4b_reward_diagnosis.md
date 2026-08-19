# Qwen3.5-4B OpenR1 reward-signal diagnosis

## Scope and safety

All experiments in this record are frozen-weight Hugging Face generation unless
explicitly called a GRPO calibration. They construct no Ray cluster, vLLM
server, FSDP worker, optimizer, gradient, or checkpoint. Each used only GPU 2
after a read-only four-GPU preflight and released it on exit.

Pinned identities remain unchanged:

- Model: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Training source: `open-r1/OpenR1-Math-220k`, revision `e4e141ec9dea9f8326f4d347be56105859b2bd68`.
- Reward: the locked upstream `math_reward`, comparing the final balanced `\boxed{...}` answer.

## What was ruled out

The original 256-token official GRPO attempt reached `global_step_1`, but its
actual OpenR1 rollout was all zero reward: none of eight completions had a
boxed answer. Its second update then remained in `actor_update_actor` without
a new log or checkpoint for more than ten minutes. It was stopped by SIGINT
only to its verified own process group; `global_step_1` was preserved and the
launcher exited with status 120. This is a failed artifact, not effective learning.

Frozen diagnostics established the prompt and length boundary:

| Contract | Samples | Boxed | Correct reward | Cap hits | Conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| concise derivation, 256 tokens | 8 | 0 | 0 | 8 | insufficient |
| concise derivation, 384 tokens | 8 | 0 | 0 | 8 | insufficient |
| concise derivation, 512 tokens | 8 | 1 | 0 | 7 | insufficient |
| concise derivation, 768 tokens | 8 | 4 | 1 | 4 | positive but sparse |
| final answer only, 256/384 tokens | 16 | 16 | 1 | 0 | format works, reasoning quality collapses |

The concise 768-token four-sample experiment had four correct samples among
32, but all were from one prompt: seven groups were `[0,0,0,0]` and one was
`[1,1,1,1]`. GRPO would still produce zero within-group advantage, so it was
rejected as a training configuration.

## Accepted calibration contract

> Solve in at most three short sentences. Do not restate the problem, describe
> a plan, or add any extra commentary. End the response with the final answer
> exactly as `\boxed{...}`.

On eight OpenR1 prompts with four samples per prompt and a 384-token cap, this
produced 30/32 boxed answers, mean response length 215.375, only 4/32 cap
hits, and one correct answer. Crucially, prompt group 1 had reward
`[0,0,0,1]`: the first observed non-degenerate GRPO group. Its raw sample
SHA-256 is `9aa3e7ed6c4022c5c442f2ed61a0a7f5a1edf683100c2062e7cef13374db2bea`.
Artifact: `qwen3.5-4b-openr1-hf-short-solution-grpo-group-20260819T1557`.

## Official one-step pass and two-step memory boundary

The official one-step artifact
`qwen3.5-4b-openr1-grpo-one-step-v5-short-20260819T1705` completed with exit status
0. Its real vLLM/FSDP2 update had reward mean 0.375, non-zero group advantages
([-1.50, 0.50]), actor loss -0.00498, grad norm 6.54 and no response-cap hit
(mean response length 116.125). This confirms the 384-token contract in the
official training path, not merely frozen HF generation.

Two-step repeats established a separate memory boundary. A GRPO update has
`train_batch_size * rollout.n` sequences, which must be divisible by agent-loop
workers. Both legal layouts (two prompts/four rollout/eight workers, and one
prompt/four rollout/four workers) completed the first update and saved
`global_step_1`, but their second actor update OOMed on GPU 0 requesting a further
1.19 GiB (about 1.1 GiB free). This is not an all-zero-reward or data-exhaustion
failure. The next isolated trial keeps the contract unchanged and turns on FSDP
actor parameter offload plus PyTorch expandable allocation segments.

## 3+1 topology as the next memory test

The two-trainer-GPU actor hits a repeatable 1.19 GiB allocation peak during its
second update.  Parameter offload lowers its idle residency but did not remove
that peak: with `expandable_segments:True`, two 44.52 GiB L20 trainer GPUs still
left only 279.75 MiB free before the same request.  This rules out treating
allocator tuning as the next likely fix.

The next run therefore used the supported asymmetric one-step-off-policy
topology: three FSDP2 trainer GPUs and one TP=1 vLLM rollout GPU. It was a new
run from the Qwen base model, never a resharded continuation of a 2+2 FSDP
checkpoint. With `train_batch_size=3` and `rollout.n=4`, each update has 12
trajectories, divisible by both three trainer GPUs and four agent workers.
Before launch, the launcher performed only a read-only four-GPU
compute-process check; no foreign process was killed or altered.

## Completed 3+1 two-step calibration

Artifact:

```text
/mnt/storage01/zhangwenchao02/repos/mini-verl-l20/artifacts/
qwen3.5-4b-openr1-grpo-two-step-v5-short-trainer3-rollout1-20260819T1820
```

It ran with locked upstream `c4b389adadc58ce51cb2b63e70df497ca166d77f`,
the existing local locked runtime (torch 2.11.0+cu130, transformers 5.5.3,
Ray 2.55.1, vLLM 0.24.0), 384 response tokens, and the new ordered six-row
slice at positions `0,1,2,3,4,5`. The slice SHA-256 is
`235906816422171ff7cba83fa340e7e146ceec0652b47b10224b28d8ad2418df`;
its audit JSON SHA-256 is
`ce6dfcc285b2ec264c635df3fb150be45304db1912c272f22cda37c656353bea`.

The run completed `global_step_1` and `global_step_2`, wrote all three model
and optimizer shards for each checkpoint, exited with status 0, and released
all of its GPU memory. TP=1 vLLM loaded, captured CUDA graphs, synchronized
weights twice, and rolled out successfully. This demonstrates that 3 FSDP2
trainer GPUs + 1 TP=1 rollout GPU removes the reproducible second-update OOM
seen with the 2+2 layout under this calibration contract.

The outcome is deliberately qualified. Step 1 had raw reward scores
`[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]` (mean 1/12), group advantages from
about -0.50 to 1.50, actor loss -0.021725, grad norm 4.0718, mean response
length 119.92 and no cap hit. It was a real non-degenerate GRPO update. Step 2
had 12/12 zero reward, zero policy-gradient loss and grad norm 0.0014; its
model/checkpoint update still completed, but it contributes no useful RL signal.
The step-2 response cap ratio was 1/3, another reason not to extrapolate this
six-row run into a quality claim.

Initial, step-1, and final frozen 64-row validation accuracy@1 were 0.625,
0.671875, and 0.65625 respectively. That is retained as an observation only:
two updates and 64 questions are not evidence of a generalization improvement.
The relevant artifact SHA-256 values are: train log
`14dd5c80b132a3fe7e98d08dfaaa8e75a6df134f7470aefd4d30a17777f3c2bd`,
step-1 rollout `01eb12e79d0a4ede356dcb06a17e4040e09cb9cc08ed0fb122a95d53d677f226`,
and step-2 rollout
`70112b61effb1f086de8a8b4f0dcf5815e9b7c6832e8db9f1fcec74e0dd8ddaa`.

## Frozen calibration inputs

| File | SHA-256 |
| --- | --- |
| `calibration-2048-openr1-math-v5-short.parquet` | `bbfa14f302d55789917df752e9f1568ea3f4f700ba9b5fe9a957d0bd1eb24e3a` |
| `calibration-math-test-64-v3-short.parquet` | `b3208ca66648232a2ef77b479c5a4b34bd242b7e99c170e715db0f83cf7527d7` |
| `calibration-two-row-v5-short.parquet` | `c387238ef2d682d1c89632a6361f9874361ab4c16c2a6ef73d8fc06c16197138` |
| `calibration-six-row-v5-short-20260819T1819.parquet` | `235906816422171ff7cba83fa340e7e146ceec0652b47b10224b28d8ad2418df` |

The two-row slice preserves converted rows 0 and 1 in order and has an audit
file recording those positions. The launcher uses `data.shuffle=False`, a 384
response cap, thinking disabled, and supports `TRAINING_STEPS=1`.

The immediate next run is not an immediate 2,048 or 10k job. It is a frozen
rollout audit to select batches with repeated within-group reward variation,
followed by a new 3+1 short training run that retains the same model, reward
and 384-token contract.

## Audited 3+1 two-step pass: both updates carried GRPO signal

The earlier six-row run proved the 3+1 memory topology but happened to sample an
all-zero second batch.  To test the distinct signal-stability question, a
frozen vLLM audit first sampled OpenR1 rows under the accepted 384-token
short-solution contract.  It retained only rows that had shown a mixed 0/1
four-sample reward group with no response-cap hit.  The audit did not update
weights; it is a selection aid, not a substitute for the training rollouts.

The resulting fresh six-row input selected original positions `0,1,24,25,26,34`
in that order (three rows per no-shuffle update).  Its parquet SHA-256 is
`76af24cb1b236b63796946b9cf17079dd8776dc81ae1157b9ab41c70a49d1fbf`; the
selection audit SHA-256 is
`fdc215802b4aec92ab45f5f4abb765de604ebc86f442ffe4f1c1a1940cb3fd57`.

Artifact:

```text
/mnt/storage01/zhangwenchao02/repos/mini-verl-l20/artifacts/
qwen3.5-4b-openr1-grpo-two-step-v5-short-trainer3-rollout1-audited-20260819T1847
```

The pinned upstream/runtime and the 3 FSDP2 trainer + 1 TP=1 vLLM rollout
topology were unchanged.  Preflight found no compute process on any L20; the
run subsequently exited `0`, saved complete world-size-3 actor and optimizer
shards at `global_step_1` and `global_step_2`, and released all four GPUs.
Total wall time was 491.78 seconds.

| Metric | Step 1 | Step 2 |
| --- | ---: | ---: |
| 12 rollout rewards | 4 × 1, 8 × 0 | 7 × 1, 5 × 0 |
| Per-prompt groups | `[0,1,0,0]`, `[0,0,0,0]`, `[1,1,0,1]` | `[1,1,0,1]`, `[0,0,1,1]`, `[1,1,0,0]` |
| Reward mean | 0.3333 | 0.5833 |
| Advantage range | -1.50 to 1.50 | -1.50 to 0.8660 |
| Actor loss / grad norm | -0.20284 / 8.1991 | 0.02760 / 9.1715 |
| Mean response length / cap ratio | 112.25 / 0 | 164.25 / 1/12 |
| Actor allocated / reserved peak (GiB) | 25.50 / 37.88 | 31.53 / 37.92 |

This is the first completed official two-step Qwen3.5-4B GRPO calibration in
which **both actual training batches** had mixed rewards and non-zero relative
advantages.  The raw rollout SHA-256 values are
`26284602bbe80edcb7082190827aad73d4a6ecd921597d13b733a169ffe5f5bb`
(step 1) and `154914510c04822ee8f5a42e3048e6c726d024c9e23b0088250f757f4b690a79`
(step 2); the train-log SHA-256 is
`af11dd155100efb548afbc831d712df907502a954040644c111790e44067a286`.

The 64-row deterministic MATH-lighteval observation was 0.640625 initially,
0.65625 after step 1, and 0.671875 after step 2.  It remains an observation,
not a quality claim: two updates, a selected six-row training slice, and 64
evaluation questions cannot establish generalization.  The legitimate next
gate is a five-step (15-row) run over the remaining audited, no-cap mixed-signal
positions before considering the fixed 2,048-row calibration or 10k subset.

## Five-step signal-stability gate

The 15-row, no-shuffle follow-up used original audited OpenR1 positions
`0,1,13,24,25,26,34,50,51,52,59,64,66,68,71`, batch size 3, four rollouts
per prompt, and the same pinned 3 FSDP2 + 1 TP=1 vLLM topology.  Input parquet
SHA-256: `58d69c813c422f7960cc4f7b98d4d43eb7bc18f634d4c458be56cb3e418c097c`.

Artifact:

```text
/mnt/storage01/zhangwenchao02/repos/mini-verl-l20/artifacts/
qwen3.5-4b-openr1-grpo-five-step-v5-short-trainer3-rollout1-audited-20260819T1857
```

It completed all five updates, saved `global_step_1` through `global_step_5`
with model and optimizer shards, exited 0, and released all GPUs.  The five
actual rollout batches had respectively 2/12, 5/12, 2/12, 6/12, and 4/12
positive rewards; their mixed-reward prompt-group counts were 2/3, 3/3, 1/3,
1/3, and 3/3.  Consequently every update had at least one usable relative
advantage group and a non-zero actor gradient norm (7.36, 8.82, 4.48, 4.77,
8.72).  Response cap ratios were 0, 0, 1/12, 1/12, and 0.

The initial and final 64-row deterministic MATH-lighteval accuracy@1 were
0.65625 and 0.640625, with a temporary 0.671875 at updates 3 and 4.  This is
not evidence of degradation or improvement: it is a tiny, selected 15-question
training slice and a 64-question diagnostic evaluation.  It *does* clear the
engineering gate for the fixed 2,048-row calibration: five consecutive stable
FSDP2/vLLM updates with no all-zero batch and no OOM.

The five-step artifact consumes 266 GiB because the calibration launcher saves
a full world-size-3 model and optimizer checkpoint after every update.  That is
correct for a short fault-localization gate but infeasible for a 683-update
(2,048-row, batch-size-3) epoch.  The launcher now parameterizes `SAVE_FREQ`
and `TEST_FREQ`; the next run uses sparse checkpoints and sparse frozen
validation while still retaining raw rollout samples for every update.

## Formal short run launched: processor-filtered 2,037 rows / 679 steps

Before the formal run, an initial 2,046-row / 682-step launch was stopped during
initialization, before any rollout or checkpoint, because upstream
`RLHFDataset` reported only 2,038 rows after its `max_prompt_length=512` filter.
The startup artifact and log are retained at
`qwen3.5-4b-openr1-grpo-2046row-682step-v5-short-trainer3-rollout1-20260819T1914`;
no unrelated process was signalled.  The cause was Qwen3.5's
conditional-generation `AutoProcessor` path: a standalone tokenizer does not
reproduce its chat-template length accounting.

The replacement input applies the exact upstream processor contract
(`apply_chat_template(..., enable_thinking=False, add_generation_prompt=True)`,
then processor-tokenizer with no added special tokens).  It excludes original
candidate rows `88,409,752,819,882,1200,1711,2037` as 533--1,855 token prompts,
selects the first 2,037 of the remaining 2,038 rows, and leaves original row
2045 for a later run.  Its parquet SHA-256 is
`acc92db0d549dc9d9996162f620df6db2d871ecf16af971a0b14fd9b84ced567`;
its audit SHA-256 is
`2b01f7053f3636cc042181593c0d45faf6dd7336456ad52c5fdabac15ab66419`.

The final formal artifact is
`qwen3.5-4b-openr1-grpo-2037row-679step-v5-short-trainer3-rollout1-20260819T1919`.
Upstream has verified `filter dataset len: 2037`, `Size of train dataloader:
679`, and `Total training steps: 679`.  It is a one-epoch, no-shuffle run with
the unchanged 3 FSDP2 + 1 TP=1 vLLM topology, `SAVE_FREQ=340`, and
`TEST_FREQ=170`; it has an initial frozen 64-row MATH-lighteval accuracy@1 of
0.65625.  At the time this record was written it had completed the first two
real updates without an error.  Final reward, checkpoint, and evaluation claims
must wait for its clean exit.
