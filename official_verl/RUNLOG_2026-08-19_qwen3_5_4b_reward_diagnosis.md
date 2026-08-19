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

## Prepared next official-verl test

| File | SHA-256 |
| --- | --- |
| `calibration-2048-openr1-math-v5-short.parquet` | `bbfa14f302d55789917df752e9f1568ea3f4f700ba9b5fe9a957d0bd1eb24e3a` |
| `calibration-math-test-64-v3-short.parquet` | `b3208ca66648232a2ef77b479c5a4b34bd242b7e99c170e715db0f83cf7527d7` |
| `calibration-two-row-v5-short.parquet` | `c387238ef2d682d1c89632a6361f9874361ab4c16c2a6ef73d8fc06c16197138` |

The two-row slice preserves converted rows 0 and 1 in order and has an audit
file recording those positions. The launcher uses `data.shuffle=False`, a 384
response cap, thinking disabled, and supports `TRAINING_STEPS=1`.

The next run is deliberately one official vLLM/FSDP GRPO step, not an
immediate two-step or 10k run. It must prove from the trainer's raw rollout
that vLLM samples still contain group reward variation and that reward,
advantage, actor loss, and gradient norm are not all zero. Only then may a
fresh two-step artifact be launched.

## Current external blocker

At 2026-08-19 17:00 CST, GPU 2 is occupied by a root-owned, unrelated
`tests/hnsw_baseline_score.py` process (PID 1376274, about 18.5 GiB). It was
not created or modified by this work. The launcher requires all four GPUs to
be empty, so no official four-GPU job is started until it exits.
