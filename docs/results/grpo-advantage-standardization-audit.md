# GRPO 组内标准差归一化审计（2026-08-22）

## 结论

正式 Qwen3.5-4B run 的每个 prompt 都有 4 个 rollout。当前 pinned VeRL 默认使用
`algorithm.norm_adv_by_std_in_grpo=true`：先减去组均值，再除以组内标准差。

对完整 679-step、2,037 个 prompt group 的真实 reward 做离线复算后，关掉标准差除法
**不会改变哪些 group 有/没有 GRPO 信号**，但会显著并且非均匀地缩小 advantage：

| outcome-level 指标 | 标准 GRPO（除 std） | 不除 std | 比例 |
|---|---:|---:|---:|
| mean absolute advantage | 0.186887 | 0.098245 | 0.526× |
| RMS advantage | 0.422576 | 0.221636 | 0.524× |
| total absolute advantage | 1522.756 | 800.500 | 0.526× |

485/2,037 个 group 的 reward 为 mixed 0/1，1,353 个全错、199 个全对 group 都仍没有
相对 policy-gradient 信号。对 mixed group，`[1,0,0,0]` 或 `[1,1,1,0]` 的标准化相对
于仅减均值约放大 2×；`[1,1,0,0]` 约放大 1.732×。因此这不是单一全局 learning-rate
缩放，而是改变不同 reward composition 的相对权重。

## 为什么值得作为下一项受控消融

- 只改一个已被上游支持的开关：`algorithm.norm_adv_by_std_in_grpo=false`；模型、训练
  rows、4-sample rollout、reward、KL、PPO-style update、3+1 GPU 拓扑均不变。
- 它比 RLOO 更有区分度：在固定 group size=4 时，RLOO 与“只减均值”只相差常数
  `4/3`；而 std normalization 的倍率会随 1/2/3 个正样本而变。
- 它不是质量结论。必须先跑 20-step calibration，检查 exit status、mixed reward groups、
  gradient norm、KL、response cap、显存与 checkpoint；通过后才能考虑 170-step 开发级
  run。

## 可复查证据

- 输入：正式 artifact 的 `rollout_samples/1.jsonl` … `679.jsonl`，8,148 条样本。
- 远端报告：`artifacts/advantage-audits/2026-08-22-grpo-vs-no-std-679step.json`；SHA-256
  `620fc09781be58b9c1e59a12c1aa841a257c940934d6d38ed0747cd08f60e0df`。
- [analyze_grpo_advantage_scaling.py](../../official_verl/analyze_grpo_advantage_scaling.py)
  复现该计算；单测覆盖 `[1,0,0,0]` 的 pinned VeRL 标准化值
  `[1.5,-0.5,-0.5,-0.5]` 与退化组为零。

## 后续 gate

只在锁定的 FSDP/vLLM runtime 恢复并通过 import/GPU preflight 后，才启动一个从 base
checkpoint 开始的 20-step no-std calibration。若其梯度或 KL 明显异常、出现 OOM、或 reward
组大面积退化，则记录失败并停止长 run；不会把短 run 的 64 题监控分数当作算法优越性的证据。
