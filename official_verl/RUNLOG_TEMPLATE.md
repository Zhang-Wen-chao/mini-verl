# Official verl GRPO run log

Copy this file into `official_verl/artifacts/<run-id>/RUNLOG.md` for every run.
The artifact directory is intentionally ignored because it can include checkpoints
and samples; commit the completed `official_verl/verl.lock.json` and any small
sanitized configuration needed to reproduce an accepted result.

## Identity

- Date and operator:
- `mini-verl` commit:
- Official `verl` commit (must match `official_verl/verl.lock.json`):
- Model snapshot / revision:
- Hardware (GPU model, count, memory, driver, CUDA):
- Python, PyTorch, and verl environment versions:

## Task and configuration

- Dataset source and immutable revision:
- Train / validation / test counts:
- Prompt template revision:
- Verifier implementation and answer-extraction examples:
- Algorithm: GRPO
- Group size, temperature, top-p, max prompt tokens, max response tokens:
- Batch / micro-batch / accumulation / learning-rate / KL settings:
- Exact upstream config path and command (copy verbatim):

## Preflight evidence

- `python official_verl/preflight.py ... --write-lock` output:
- `python official_verl/preflight.py ... --require-lock` output:
- Base-model held-out score:
- First rollout reward histogram (must not be all 0 or all 1):

## Results

- Checkpoint paths and restore test:
- Training metrics: reward, KL, response length, loss, throughput:
- Post-training held-out score using the identical verifier:
- Representative successes, failures, and suspected reward-hacking behavior:
- Decision: reject / rerun / promote to formal 4B / optional 8B scale-up:
