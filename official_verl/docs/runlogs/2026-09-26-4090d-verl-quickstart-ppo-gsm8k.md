# Official verl quickstart: PPO on GSM8K with Qwen2.5-0.5B on 4090D GPUs

## Outcome

Upstream's published quickstart completed twice on the 4090D box — once on one
GPU (29 steps, 20 m 33 s) and once on four (29 steps, 11 m 12 s), both exit
status `0`, both with validation points. The recipe itself needed **no
adaptation**: Qwen2.5-0.5B-Instruct and the published batch sizes fit 24 GiB
with room to spare, and upstream's "use micro-batch 1 below 32 GB HBM" advice
was not required. Every failure mode encountered was in the container image, not
in the training configuration.

This is a **systems result, not a quality result**, and the headline curve is
the reason. The base model scores `0.000758` (1/1319) on the held-out GSM8K
split under this recipe's prompt and greedy validation decoding; it reaches
0.45–0.50 within 29 steps at learning rate `1e-6`. Over that same window mean
response length *halves* (315 → 170 tokens) and the response cap ratio collapses
(0.152 → 0.012). The rise is dominated by responses becoming **parseable**, not
by reasoning improving. Upstream's quickstart sets `trainer.val_before_train=False`,
so a verbatim copy of it structurally cannot observe this; the base-model anchor
had to be measured by a separate run.

## Identity

- Date: 2026-09-26 UTC.
- Official verl: tag `v0.7.1`, commit `bec9ef74768dd201881cd4e54cd0385e87caae27`
  (2026-03-16).
- Container image: `verlai/verl:vllm017.latest`, digest
  `sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6`.
  Base `nvidia/cuda:12.9.1-devel-ubuntu22.04`; Python 3.12.13, PyTorch
  `2.10.0+cu129`, vLLM `0.17.0`, Ray `2.54.0`. The image ships dependencies
  only — `verl` itself is a caller-supplied checkout.
- Model: `Qwen/Qwen2.5-0.5B-Instruct` by HF repository id. **The weight revision
  is not captured — this is a gap**, and upstream's own preflight/lock convention
  would require pinning it before any result is treated as reproducible.
- Data: GSM8K through upstream's own `examples/data_preprocess/gsm8k.py`
  (`openai/gsm8k`, `main`); 7,473 train / 1,319 test rows.
  `train.parquet` SHA-256
  `8fb13a0cd8621e5fcacdee3ffe28b3a369975478ea7ab514781f3f190cba2488`;
  `test.parquet` SHA-256
  `0dcd50ed32caa9d8f434d4e3ef8031bc3e49b0f823655a53be5c002e727f17dd`.
- Hardware: 4 × GeForce RTX 4090D (24 GiB each), driver 550.127.05, no NVLink.
  Shared host: GPUs 1–3 carried unrelated long-running resident processes
  (612 / 612 / 4,347 MiB), so the 1-GPU run pinned `CUDA_VISIBLE_DEVICES=0` and
  the 4-GPU run took all four.
- Working directory, container name and jump host are recorded behind
  placeholders (`<workdir>`, `<container>`, `<relay-host>`) consistent with the
  repository's desensitization convention.

## Task and exact launch

PPO through upstream `verl.trainer.main_ppo`, FSDP actor and critic, vLLM
rollout, KL-in-reward at `kl_coef=0.001`. Fields copied unchanged from the
published command:

- `data.train_batch_size=256`, `actor.ppo_mini_batch_size=64`,
  `actor.ppo_micro_batch_size_per_gpu=4`,
  `critic.ppo_micro_batch_size_per_gpu=4`
- `data.max_prompt_length=512`, `data.max_response_length=512`
- `actor.optim.lr=1e-6`, `critic.optim.lr=1e-5`
- `rollout.name=vllm`, `rollout.tensor_model_parallel_size=1`,
  `rollout.gpu_memory_utilization=0.4`,
  `rollout.log_prob_micro_batch_size_per_gpu=8`,
  `ref.log_prob_micro_batch_size_per_gpu=4`
- `trainer.save_freq=10`, `trainer.test_freq=10`
- `trainer.total_epochs=1` (upstream ships 15 = 435 steps; at the measured
  36–43 s/step that is ≈4.5 h)

Launchers: `official_verl/run_qwen2_5_0_5b_1gpu_quickstart.sh` and
`official_verl/run_qwen2_5_0_5b_4gpu_quickstart.sh`. Both wrap the same command
with this repository's standard durable log and `logs/exit_status` capture.

Five deviations from the published command, and nothing else:

1. `CUDA_VISIBLE_DEVICES=0` on the 1-GPU run — shared machine.
2. Data paths supplied by the caller instead of `$HOME/data/gsm8k`.
3. `MODEL_PATH` overridable; defaults to the upstream repository id.
4. `trainer.total_epochs=1` instead of 15.
5. **4-GPU run only:** `trainer.val_before_train=True`. Upstream ships `False`.
   This is the control, not a tuning knob — see Interpretation.

## Preflight evidence

- Base-model held-out score (4-GPU run, step 0 with `val_before_train=True`):
  `val-core/openai/gsm8k/acc/mean@1 = 0.000758150113722517`, i.e. **1 of 1319**.
- The reward distribution is non-degenerate on every step: `critic/score/max`
  is `1.0` and the minimum is `0.0`, so within-group variance — and with it the
  advantage and `actor/grad_norm` (1.9–3.6) — never collapses to zero. A group
  that is all-zero or all-one would produce `grad_norm=0` legitimately and the
  run would be uninformative; this one is not that case.

## Results

### One GPU — 29 steps, 20 m 33 s, 36–43 s/step

| step | resp len | cap ratio | train reward | actor grad | critic grad | entropy | val acc@1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 315.0 | 0.152 | 0.0078 | 1.89 | 300.4 | 0.493 | — |
| 5 | 290.4 | 0.082 | 0.0039 | 1.88 | 49.0 | 0.471 | — |
| 10 | 269.7 | 0.070 | 0.0391 | 3.32 | 42.8 | 0.486 | 0.0220 |
| 15 | 225.8 | 0.039 | 0.2305 | 3.01 | 28.2 | 0.444 | — |
| 20 | 188.5 | 0.023 | 0.3828 | 3.26 | 60.6 | 0.408 | 0.4268 |
| 24 | 162.9 | 0.008 | 0.5000 | 3.64 | 40.6 | 0.339 | — |
| 29 | 170.4 | 0.012 | 0.4805 | 3.55 | 100.4 | 0.304 | 0.4973 |

`actor/kl_loss` is `0.0000` on every step. `critic/vf_explained_var` is
`-2679.55` at step 1 — an unfitted value head, not a fault; do not chase it.

### Four GPUs — 29 steps, 11 m 12 s, 17–19 s/step

| step | resp len | cap ratio | train reward | actor grad | critic grad | entropy | val acc@1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | — | — | — | — | — | — | **0.000758** |
| 1 | 308.8 | 0.121 | 0.0117 | 2.09 | 314.7 | 0.479 | — |
| 10 | 314.8 | 0.102 | 0.0469 | 2.11 | 80.6 | 0.495 | 0.02350 |
| 20 | 286.1 | 0.090 | 0.2930 | 2.28 | 61.4 | 0.461 | 0.38362 |
| 29 | 259.5 | 0.035 | 0.4805 | 2.44 | 47.9 | 0.292 | 0.45186 |

### Resource envelope

- 1 GPU: `perf/max_memory_allocated_gb` 16.13, reserved 19.67. GPU utilization
  75–78%.
- 4 GPUs: allocated 6.47, reserved 10.26 per GPU. Utilization 73% on GPU 0 and
  89–99% on GPUs 1–3.
- Host memory `perf/cpu_memory_used_gb`: **113 GB on 1 GPU, 130 GB on 4.** This
  is the tighter constraint on this box, not VRAM — the quickstart's own
  recommendations only ever discuss HBM.
- Four GPUs delivered **1.83×** over one, not 4×: `train_batch_size=256` is
  fixed, so each rank sees 64 samples while four actor/critic/ref copies and
  four vLLM rollout instances add synchronization. A 0.5B model cannot fill four
  cards at this batch size; raising the batch is a precondition for measuring
  scaling, not a tuning detail.
- Weight sync per step (`update_weights done`) took 0.36–0.42 s in both runs.

## The rising curve is mostly format elicitation

Three observations, in descending order of strength:

1. **Response length collapses monotonically while the cap ratio collapses with
   it.** 315 → 170 tokens (1 GPU) and 309 → 259 (4 GPUs); `response_length/clip_ratio`
   0.152 → 0.012. At step 1, roughly one response in seven ran into the 512-token
   cap or otherwise never reached a well-formed answer, so upstream's rule reward
   could not extract a final answer and scored it zero. What training first
   changes is "answer within the budget".
2. **The transition is abrupt, not gradual.** Aggregate train reward moves
   0.0078 → 0.4805, but the rise is concentrated between steps 12 and 16
   (0.105 → 0.320) — a sigmoid, not the incremental curve a capability gain
   would produce. Format becoming reachable is a switch; skill is a ramp.
3. *(Weak, do not lean on this.)* The endpoint 0.4973 is close to the widely
   reported GSM8K figure for Qwen2.5-0.5B-Instruct. This is consistent with
   "the capability was already there and got elicited" but is **not** evidence:
   that figure comes from an unrelated evaluation configuration, the number is
   not verified here, and the coincidence may be just that.

**Interpretation.** For a rule reward that requires a final answer marker
combined with a hard `max_response_length`, the first gradient signal available
is "emit the answer sooner". The measured curve is therefore largely a
format/elicitation phase transition. This does **not** prove reasoning did not
also improve; separating the two needs a count of responses containing the
answer marker, which this run did not collect. Note that upstream ships
`max_response_length=512`, which is tight for a 0.5B model and is what makes the
format phase dominate.

This is the same failure family as
[坑 1 in the L20 lessons record](../../../docs/operations/l20-lessons-learned.md):
there, greedy decoding on a hard problem never emits `<eos>` and the evaluation
silently measures a token cap; here, responses never reach the answer marker and
the reward silently measures formatting. In both cases the metric is about
output shape rather than capability.

## Environment blockers

All three are image/container level; none is in the training recipe.

1. **Upstream images moved to CUDA 13 at v0.8.0, which this GPU class cannot
   run.** From `v0.8.0` on, `docker/Dockerfile.stable.vllm` builds on
   `nvidia/cuda:13.0.2-devel-ubuntu24.04` and the config blob pins
   `NVIDIA_REQUIRE_CUDA=cuda>=13.0`. Driver 550.127.05 tops out at CUDA 12.4,
   and **GeForce cards do not support forward compatibility**, so this is a dead
   end rather than a configuration problem. `v0.7.1` is the last usable release
   here. Cost of learning this from a pull instead of from the source: 13 GiB.
   Cheaper check: read the `FROM` line of `docker/Dockerfile.stable.vllm` at the
   tag, without pulling.
   *Cross-machine contrast that localizes the cause:* the L20 box runs the same
   550.127.05 driver and executes `PyTorch 2.11.0+cu130` fine
   ([2026-08-19 clean rerun](2026-08-19-qwen3-0.6b-gsm8k-clean-rerun.md)) — L20
   is a datacenter part and does support forward compatibility. Same driver,
   different GPU class, opposite outcome.
2. **CUDA Error 804 in the container.** The image carries two `libcuda`
   libraries and `libcuda.so.1` resolves to the image's forward-compat build
   rather than the bind-mounted host driver. GeForce rejects forward
   compatibility, so `cudaGetDeviceCount` fails with
   `Error 804: forward compatibility was attempted on non supported HW`.
   Fix, inside the container:
   ```bash
   ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.550.127.05         /usr/lib/x86_64-linux-gnu/libcuda.so.1
   ln -sf /usr/lib/x86_64-linux-gnu/libcudadebugger.so.550.127.05 /usr/lib/x86_64-linux-gnu/libcudadebugger.so.1
   ldconfig
   ```
   This is a container writable-layer change and does not survive a rebuild.
   Occasional reports of `ln -sf` appearing not to take effect were resolved by
   running it a second time.
3. **The registry mirror parked inside one 4.4 GiB layer.** The pull stalled
   partway through a single large layer and never progressed: the progress
   display kept repainting and `elapsed` kept climbing, but the content store's
   disk usage stopped growing and no error or retry ever appeared. Two different
   mirrors reproduced it. Because a containerd content store is
   content-addressed, the layer can be finished out-of-band and installed
   directly: resume the partial bytes from `ingest/` with
   `curl -C - --retry-all-errors`, verify SHA-256 against the layer digest, then
   install into `blobs/sha256/<digest>` and re-run the pull, which then resolves
   the layer as `exists`. `official_verl/resume_stalled_blob.sh` automates this.
   Diagnostic rule: judge a stall by **content-store disk growth, not the
   progress bar**.
4. **HF Xet redirection.** The model download only completes against the real
   HF endpoint; the mirror 302-redirects Xet-backed weights to a host it does
   not mirror, stalling the large safetensors file while small tokenizer/config
   files succeed — which makes it look like a slow network.

### Upstream documentation that is now stale

- The quickstart's sample log shows `timing/gen`, `timing/ref`,
  `timing/update_critic` fields. **No `timing/*` fields exist in v0.7.1**; the
  current names are `perf/*` and `global_seqlen/*`. Following that sample
  literally leads to concluding the wrong thing ran.
- The quickstart's install note and the repository's own
  `official_verl/bootstrap_verl.sh` still reference
  `https://github.com/volcengine/verl.git`. **The project has moved to
  `verl-project/verl`**; the old path redirects today but should be updated.
- The documented evaluation metric name is `val/test_score/openai/gsm8k`; v0.7.1
  actually reports `val-core/openai/gsm8k/acc/mean@1`.

## Decision

- **Accept as a systems record**: the official PPO + GSM8K + vLLM path runs
  end-to-end on consumer 4090D hardware with no recipe changes, on 1 and 4 GPUs,
  at the pinned revision and image digest above.
- **Do not promote as a quality or algorithm result.** 29 steps of a 0.5B model
  with a format-dominated reward curve supports no comparison against any other
  configuration, including mini-verl's own GRPO runs.
- **Adopt `trainer.val_before_train=True` for any quickstart-derived evaluation
  claim.** Without the base-model anchor, the 0.0008 → 0.50 transition reads as a
  ~600× improvement; with it, the same numbers read as elicitation. Upstream's
  default hides the anchor and the run log is where it must be recovered.
- **Open gaps**, in the order they should be closed: pin the model weight
  revision; record the response-marker rate to test the elicitation account
  directly; re-run with the batch scaled with GPU count before making any
  throughput-scaling statement.
