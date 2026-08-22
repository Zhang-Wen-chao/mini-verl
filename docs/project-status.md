# 项目状态：mini-verl / official-verl-grpo

更新：2026-08-22。分支：`official-verl-grpo`。

## 一句话结论

这个分支已经不只是“计划做一个 GRPO 框架”：它同时完成了一个可验证的最小 GRPO
实现，以及一次官方 `verl` 上的 Qwen3.5-4B 真实训练。最强证据是独立 MATH held-out
200 题的对照结果是：**未做 GRPO 强化训练的原始 Qwen3.5-4B base checkpoint** 为
**5/200（2.5%）**；**679-step GRPO checkpoint** 为 **17/200（8.5%）**，即
**绝对 +6.0 个百分点**、相对 **3.4×**。这 200 题只用于前后评测，不参与更新。

这说明当前训练契约在该固定模型、数据、奖励、算力和评测协议下有效；它不是对所有
数学任务、所有模型或更大训练规模的泛化承诺。

## 已完成的主线

```text
mini_verl：用小而可测的实现验证 GRPO 语义和系统边界
    ↓
official_verl：在锁定的官方训练栈上先完成 0.6B 系统 smoke
    ↓
Qwen3.5-4B：校准奖励、解决 2+2 拓扑 OOM、验证 3+1 拓扑
    ↓
2037 训练题 / 679 step：完成真实 GRPO 正式 run
    ↓
独立 MATH held-out：2.5% → 8.5%，得到可复查的质量提升证据
    ↓
170-step no-std / standard GRPO：完成单 seed 开发级受控对照；保留但不夸大其结论
    ↓
PPO/GAE + 4B Critic vs standard GRPO：同 12-trajectory 预算的 5-step gate 均完成
```

| 层级 | 状态 | 做成了什么 | 应该读什么 |
|---|---|---|---|
| 最小框架 `mini_verl/` | 已完成核心闭环 | trajectory 契约、GRPO + KL、HF rollout/trainer、策略同步、checkpoint、DDP smoke、长度调度与性能观测 | [框架架构与实现进度](architecture/mini-verl-architecture.md)、[运行指南](guides/runbook.md) |
| 官方训练系统 `official_verl/` | 已完成 | 锁定官方 `verl`、FSDP2 + vLLM、preflight、数据转换、规则奖励、checkpoint 与 clean exit | [官方实验资产索引](../official_verl/README.md)、[0.6B 系统 smoke](../official_verl/docs/results/qwen3-0.6b-gsm8k-smoke.md) |
| 4B 训练质量 | 已完成第一条有效实验 | Qwen3.5-4B、2037 训练题、679 step、4×L20、独立 held-out 正向提升 | [679-step 结果](results/qwen3.5-4b-grpo-679-step.md) |
| GRPO 算法开发对照 | 已完成单 seed 筛选 | 先通过 20-step no-std health gate，再从 base 跑 no-std / standard 各 170 step；两段均 clean exit、170 rollout、完整 checkpoint | [170-step 开发对照](results/qwen3.5-4b-grpo-170-step-development-ablation.md) |
| 4B PPO actor-critic 可行性 | 已完成 5-step gate | 真实 PPO/GAE + 4B Critic 与 standard GRPO 在同一 12-trajectory/step 契约下各完成 5 update、checkpoint 与 clean exit；PPO 借助 Critic activation offload 通过完整 batch 的显存边界 | [PPO/GRPO 5-step 公平对照](../official_verl/docs/runlogs/2026-08-22-qwen3.5-4b-ppo-grpo-fair-development-comparison.md) |
| 评测与实验可靠性 | 已识别并修复关键问题 | 训练/评测去重、答案格式归一化、逐题落盘、单题超时、评测回落诊断 | [回落分析](results/step-510-to-679-regression-analysis.md)、[经验记录](operations/l20-lessons-learned.md) |

## 当前最值得展示的亮点

### 1. 不是只跑通，而是有 held-out 质量证据

| 实验 | 训练 | 评测 | 结果 | 结论 |
|---|---|---|---|---|
| Qwen3.5-4B GRPO 正式 run | OpenR1-Math 过滤后 2037 题；679 step；4×L20 | MATH-lighteval test 随机 200 题；训练未见；**未做 GRPO 的 base** 与 **679-step checkpoint** 使用同 prompt、greedy 和归一化评分 | 未做 GRPO：5/200（2.5%）→ GRPO 679 step：17/200（8.5%） | **+6.0pp、3.4×**，可作为“该训练设置产生 held-out 改善”的证据 |
| 训练中固定监控 | 同上 | 64 题、训练集零重叠；在训练中定期评测 | base 42/64（65.6%）→ step 510 最高 56/64（87.5%），**+21.9pp**；step 679 原始 53/64，复核约 55/64 | 证明训练中存在强学习信号；因被反复评测，最终泛化结论仍以独立 200 题为主 |

完整配置、checkpoint、评测节点、训练信号和边界见 [679-step 结果](results/qwen3.5-4b-grpo-679-step.md)。

### 2. 标准 GRPO 与 no-std GRPO 已完成一次健康的开发级对照

本次唯一预定的算法变量是 `algorithm.norm_adv_by_std_in_grpo`。在相同 2,037 条
训练数据、4×L20 3+1 拓扑、rollout n=4、legacy math reward、LR `1e-6`、reference
KL `0.001` 和 170 step 契约下：no-std 的固定 64 题 monitor 为 42/64 → 47/64；
standard 为 41/64 → 48/64。二者都完成 170 rollout、`global_step_170`、完整数值
health gate。

这只说明两种配置在真实链路上都可运行，而本次 development monitor 中 standard 的增幅
略高。由于单 seed、起点并不相同且 monitor 被反复查看，**不能据此宣称 standard
GRPO 更优**；更不能触碰 MATH held-out 200 做模型选择。细节见[170-step 开发对照](results/qwen3.5-4b-grpo-170-step-development-ablation.md)。

### 3. 对失败和负结果也留了证据

- 0.6B / 16-step 官方 smoke 被明确标为**系统成功、质量结论拒绝**，没有把它包装成提升。
- 2 trainer + 2 rollout 的 4B 拓扑第二次 actor update OOM；通过 3 trainer + 1 rollout
  的验证解决了该资源边界，而不是无依据地继续扩训。
- 510 → 679 的表面回落被逐题核查，确认其中两题是分数/小数等价答案被字符串评分器误杀。
- 变长批处理和 prefetch 的性能结论同时记录收益与代价，例如 token budget 可能降低吞吐，
  双卡 prefetch 在生成主导 workload 中只能隐藏有限 rollout 时间。

### 4. PPO 已通过完整样本预算的可行性 gate，但尚无质量胜负

PPO/GAE + 4B Critic 与 standard GRPO 都以相同的 3 prompts × 4 rollouts = 12
trajectories/step、相同数据/seed/长度/拓扑完成了 5 step：PPO 的 64 题开发 monitor 为
41/64 → 45/64，GRPO 为 40/64 → 42/64。两边初始 monitor 不同、只有单 seed 和 5 step，
所以这不是“PPO 更好”的证据。

它给出的可靠工程结论是：完整 12-trajectory PPO 首次在 Critic backward OOM；保持其余契约
不变并启用 Critic activation offload 后，PPO 可 cleanly 完成。PPO 训练 loop 为 8m35s、最后
一步 Actor 最大 allocated GPU memory 35.70 GB；匹配 GRPO 为 6m42s、31.53 GB。即在本次
短运行中，PPO 的真实 actor--critic 路径可行但资源代价更高。完整数据见
[PPO/GRPO 5-step 公平对照](../official_verl/docs/runlogs/2026-08-22-qwen3.5-4b-ppo-grpo-fair-development-comparison.md)。

### 5. mini 框架的价值在“能验证和定位”，不是重复造完整 verl

`mini_verl/` 刻意不复制 Ray、多机编排或生产级服务系统。它把下列边界拆开并加以测试：

```text
rollout → trajectory / old logprob → reward → group advantage
        → GRPO + reference KL → optimizer update → policy-version sync
```

这使得算法正确性、策略陈旧性、padding、长度准入、checkpoint 和 rollout/train
重叠可以分别验证，而不是都隐藏在一次大规模训练里。

## 读什么，不读什么

- 想看**现在和下一步**：只读本页。
- 想看**最终 held-out 实验结论**：读 [679-step 结果](results/qwen3.5-4b-grpo-679-step.md)。
- 想看**PPO 与 GRPO 的最新可行性/资源对照**：读 [PPO/GRPO 5-step 公平对照](../official_verl/docs/runlogs/2026-08-22-qwen3.5-4b-ppo-grpo-fair-development-comparison.md)，不要把它当最终质量结论。
- 想看**此前 GRPO 的算法筛选**：读 [170-step 开发对照](results/qwen3.5-4b-grpo-170-step-development-ablation.md)，不要把它当最终泛化结论。
- 想看**历史排障过程**：从 [官方实验资产索引](../official_verl/README.md) 进入 `docs/history/` 或 `docs/runlogs/`。这些是证据归档，**不是当前 roadmap**。
- 想看完整文档分类：读 [文档导航](README.md)。

## 当前边界：哪些还没有做

| 项目 | 状态 | 原因 / 位置 |
|---|---|---|
| LLM PPO actor-critic / GAE 的质量对照 | 尚未完成 | 已完成与 GRPO 匹配的 5-step 稳定性/资源 gate；仍需新的 development/final split、20-step 与多 seed，不能以 64 题 monitor 选算法 |
| 学习型 Reward Model、DPO、KTO | 未做 | 当前真实实验使用可审计的数学规则 reward，避免把奖励模型误差与 GRPO 混在首条质量结论中 |
| vLLM / SGLang 作为 `mini_verl` rollout backend | 未做 | 最小框架已有 HF backend；官方验证已使用 vLLM，下一步再决定是否抽象回接 |
| 多机、异构硬件、生产级容错 | 未做 | 有意不纳入教学/验证型最小实现 |
| 多 seed 的算法对照 | 待做 | 170-step 单 seed 只提供筛选线索；需先固定新的 development / final 划分 |
| formal-10k 对照实验 | 暂不启动 | 先决定算法变量与评测划分，再扩大数据规模，避免耗 GPU 做不可解释的长跑 |

## 下一步

1. **冻结已完成证据。** 679-step held-out、170-step GRPO 与 PPO/GRPO 5-step 配对 artifact
   均已完成；不删除 checkpoint，不把 64 题 monitor 或 MATH held-out 200 再用于随意调参。
2. **固定新的评测划分，再做 20-step。** 预先写死 development/final protocol、seed 数、步数
   和停止规则；之后以当前同 12-trajectory 契约完成 20-step PPO/GRPO development 对照。
3. **只让通过 20-step 的候选进入多 seed 或长跑。** 同时以固定样本预算与固定 GPU-hours
   两种口径报告，才讨论质量/成本权衡。
4. **再考虑数据规模或 reward / length 变量。** 只有算法筛选与评测协议清楚后，才做
   formal-10k；训练后期长度接近 cap 的问题也应作为独立变量处理。
5. **回写最小实现。** 只把已在官方链路中验证的 `dataset → rollout → verifier reward →
   group advantage → logprob → update → metrics` 语义映射回 `mini_verl`，不复制完整生产编排。

## 验证状态

2026-08-22：20-step calibration、no-std 170-step、standard 170-step、PPO 5-step 与匹配
GRPO 5-step 的训练及 watcher exit status 均为 0；PPO/GRPO 两腿各含 5 rollout 与
`global_step_5`。本地新增的 preflight / development-pair summary 定向单测均通过；完整
unittest 总数仍以最近一次 106 passed、28 skipped 的全量记录为准。
