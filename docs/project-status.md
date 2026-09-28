# 项目状态：mini-verl / official-verl-grpo

更新：2026-09-28。分支：数学 GRPO 主线在 `official-verl-grpo`；
SWE / agentic RL 实验在 `swe-grpo-l20`（本分支）。

## 一句话结论

这个分支已经不只是“计划做一个 GRPO 框架”：它同时完成了一个可验证的最小 GRPO
实现，以及一次官方 `verl` 上的 Qwen3.5-4B 真实训练。最强证据是独立 MATH held-out
200 题的对照结果是：**未做 GRPO 强化训练的原始 Qwen3.5-4B base checkpoint** 为
**5/200（2.5%）**；**679-step GRPO checkpoint** 为 **17/200（8.5%）**，即
**绝对 +6.0 个百分点**、相对 **3.4×**。这 200 题只用于前后评测，不参与更新。

这说明当前训练契约在该固定模型、数据、奖励、算力和评测协议下有效；它不是对所有
数学任务、所有模型或更大训练规模的泛化承诺。

另一条线（`swe-grpo-l20`）是 **SWE-bench 上的 agentic RL**，它给出的是一条
**可判定的负结果**：200 步训练让模型把训练池那 55 道题从 `reward=1` 13.9% 背到 48.8%，
但在零重叠的 65 道留出题上**没有任何迁移**（40 步复测 McNemar p = 1.0000）。
它的价值不在分数，而在于干净地分离了**记忆与泛化**、并定位了四处会让结论**静默失效**
的工程陷阱。详见[下文第 5 点](#5-swe--agentic-rl一条可判定的负结果而不是一个刷分数字)。

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
| 最小框架 `mini_verl/` | 已完成核心闭环 | trajectory 契约、GRPO + KL、DPO 在线偏好闭环、HF rollout/trainer、策略同步、checkpoint、DDP smoke、长度调度与性能观测 | [框架架构与实现进度](architecture/mini-verl-architecture.md)、[运行指南](guides/runbook.md) |
| 官方训练系统 `official_verl/` | 已完成 | 锁定官方 `verl`、FSDP2 + vLLM、preflight、数据转换、规则奖励、checkpoint 与 clean exit | [官方实验资产索引](../official_verl/README.md)、[0.6B 系统 smoke](../official_verl/docs/results/qwen3-0.6b-gsm8k-smoke.md) |
| 4B 训练质量 | 已完成第一条有效实验 | Qwen3.5-4B、2037 训练题、679 step、4×L20、独立 held-out 正向提升 | [679-step 结果](results/qwen3.5-4b-grpo-679-step.md) |
| GRPO 算法开发对照 | 已完成单 seed 筛选 | 先通过 20-step no-std health gate，再从 base 跑 no-std / standard 各 170 step；两段均 clean exit、170 rollout、完整 checkpoint | [170-step 开发对照](results/qwen3.5-4b-grpo-170-step-development-ablation.md) |
| 4B PPO actor-critic 可行性 | 已完成 5-step gate | 真实 PPO/GAE + 4B Critic 与 standard GRPO 在同一 12-trajectory/step 契约下各完成 5 update、checkpoint 与 clean exit；PPO 借助 Critic activation offload 通过完整 batch 的显存边界 | [PPO/GRPO 5-step 公平对照](../official_verl/docs/runlogs/2026-08-22-qwen3.5-4b-ppo-grpo-fair-development-comparison.md) |
| 评测与实验可靠性 | 已识别并修复关键问题 | 训练/评测去重、答案格式归一化、逐题落盘、单题超时、评测回落诊断 | [回落分析](results/step-510-to-679-regression-analysis.md)、[经验记录](operations/l20-lessons-learned.md) |
| SWE / agentic RL（`swe-grpo-l20`） | 已完成一条**可判定的负结果** | 2×L20 + Qwen3.5-4B + slime(Megatron+SGLang+Ray) + `mini-swe-agent`，SWE-bench Lite 上 200 步 GRPO：训练池 `reward=1` 13.9%→48.8%，零重叠留出池 65 题无迁移（40 步复测 p = 1.0000）；并定位四处**静默失效**的工程陷阱 | [SWE-GRPO on 2×L20](results/swe-grpo-l20-negative-result.md) |

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

### 5. SWE / agentic RL：一条**可判定**的负结果，而不是一个刷分数字

在 2×L20 上用 slime（Megatron + SGLang + Ray）+ `mini-swe-agent` 跑通 SWE-bench 的
agentic RL 全链路，200 步 GRPO（Qwen3.5-4B）：

| 口径 | 起点 | 终点 | 结论 |
|---|---|---|---|
| 训练池 55 题（反复看） | `reward=1` 13.9% | **48.8%** | 上升是真的 |
| 零重叠留出池 65 题 | 5.8% | 6.2%（40 步复测） | **无迁移**，题级两者都是 7/65，McNemar **p = 1.0000** |

这是**记忆 vs 泛化**的一次干净分离：同一个权重在反复看的题上大幅上升，在没见过的题上
纹丝不动。**这个结论是“可判定”的，不是“读不出”** —— 最初那轮评测跑在冒烟档 15 步上，
存在“预算不够所以测不出”这个替代解释；随后用 40 步复测把它排除掉了（基座侧
15→40 一动没动，训练侧放开到 40 步也没起来）。

**主要产出不是这个结论，而是四处「静默失效」**：冒烟值漏进生产、数据谱系没验
（120 题全部来自 SWE-bench Lite 的 `test` split、95% 是 django）、配对设计配了非配对检验，
以及**跑实验的工装自己的判据也错了**。四处都不报错、无法从日志看出，但都足以让一个
看起来干净的结论失去前提。完整边界、取证与反面教训见
[SWE-GRPO on 2×L20](results/swe-grpo-l20-negative-result.md)。

### 6. mini 框架的价值在“能验证和定位”，不是重复造完整 verl

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
- 想看**agentic RL（SWE-bench）的结论**：读 [SWE-GRPO on 2×L20](results/swe-grpo-l20-negative-result.md)。它是**负结果**，价值在于分离了记忆与泛化、并定位了四处静默失效，不是一个分数。
- 想看**历史排障过程**：从 [官方实验资产索引](../official_verl/README.md) 进入 `docs/history/` 或 `docs/runlogs/`。这些是证据归档，**不是当前 roadmap**。
- 想看完整文档分类：读 [文档导航](README.md)。

## 当前边界：哪些还没有做

| 项目 | 状态 | 原因 / 位置 |
|---|---|---|
| LLM PPO actor-critic / GAE 的质量对照 | 尚未完成 | 已完成与 GRPO 匹配的 5-step 稳定性/资源 gate；仍需新的 development/final split、20-step 与多 seed，不能以 64 题 monitor 选算法 |
| 学习型 Reward Model、KTO | 未做 | 当前真实实验使用可审计的数学规则 reward，避免把奖励模型误差与 GRPO 混在首条质量结论中 |
| DPO 的真实偏好数据与质量对照 | 未做 | `mini_verl` 已实现规则 reward 组内配对的在线 DPO 闭环（loss 双实现对拍 + HF trainer + smoke）；离线偏好数据集加载与 LLM 质量实验未做 |
| vLLM / SGLang 作为 `mini_verl` rollout backend | 未做 | 最小框架已有 HF backend；官方验证已使用 vLLM，下一步再决定是否抽象回接 |
| 多机、异构硬件、生产级容错 | 未做 | 有意不纳入教学/验证型最小实现 |
| 多 seed 的算法对照 | 待做 | 170-step 单 seed 只提供筛选线索；需先固定新的 development / final 划分 |
| formal-10k 对照实验 | 暂不启动 | 先决定算法变量与评测划分，再扩大数据规模，避免耗 GPU 做不可解释的长跑 |
| SWE / agentic RL 的**迁移** | 未达成（可判定的负结果） | 200 步训练只产生记忆：训练池 13.9%→48.8%，零重叠留出池无变化（40 步复测 p = 1.0000）。下一步要么换题源（当前 120 题全部来自 SWE-bench Lite `test` split、95% django），要么换成**基座可解 30–50% 的题池**（现在留出池题级天花板只有 7/65 = 10.8%，离地板太近） |

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
6. **SWE / agentic RL 的下一步（尚未选定）。** 40 步复测已完成，「预算不够」这个替代解释
   已排除，负结论定稿。接下来在三条里选：**(A)** 现有 55 题里筛掉 11 道从未成功过的死题
   重训，测「无梯度题稀释」占多少；**(B)** 换一个基座可解 30–50% 的题池（要新拉镜像，
   磁盘与 GPU 都有约束），这是目前最有希望的方向；**(C)** 先停在当前负结果上，把表述钉死。
   发车前的动作预算断言已落地（`SWE_AGENT_STEP_LIMIT < 40` 直接拒绝启动）。

## 验证状态

2026-08-22：20-step calibration、no-std 170-step、standard 170-step、PPO 5-step 与匹配
GRPO 5-step 的训练及 watcher exit status 均为 0；PPO/GRPO 两腿各含 5 rollout 与
`global_step_5`。本地新增的 preflight / development-pair summary 定向单测均通过；完整
unittest 总数仍以最近一次 106 passed、28 skipped 的全量记录为准。

2026-09-01：`mini_verl` 新增在线 DPO 闭环（`algorithms/dpo.py` reference/torch 双实现、
`preference.py` 规则 reward 组内配对、`HuggingFaceDpoTrainerWorker`、`examples/hf_dpo_smoke.py`）。
macOS 无 ML 依赖环境 158 passed / 40 skipped；L20 `<container>` 容器
（CUDA_VISIBLE_DEVICES=1）全套 158 passed / 0 failed，含 float64 torch 对拍、真实 GPT-2
trainer、pair 级 micro-batch 一致性与 Controller 集成测试；随后用本地 Qwen3-0.6B-Base 完成
1 次 DPO 迭代 smoke（4 trajectories、mean_reward 0.5、初始 loss≈log 2、margin 0——初始
policy 与 reference 相同的预期行为）。该 smoke 只验证管线，不是质量结论。验证在 agent 创建的
隔离目录 `repos/mini-verl-dpo-verify-20260901/` 完成，未改动 `mini-verl-l20` 副本。

2026-09-28：SWE / agentic RL（`swe-grpo-l20`）的 40 步复测完成，10/10 单元全部完整。
配对读数 i199 32/520 vs base 30/520、题级 7/65 vs 7/65、判别对 1:1、McNemar p = 1.0000；
基座侧 15→40 也一动没动（30/520 → 30/520，p = 1.0000）。⇒ 「预算不够所以测不出」
这个替代解释被排除，**负结论定稿**。同时从 200 个 rollout dump 重算训练池可学带：
797 组中全败 39.8% / 混合（有梯度）46.0% / 全胜 14.2%，55 道题里 44 道（80%）出过梯度、
11 道（20%）从未成功。文档侧修正了两处自身判据（评测 run 的 judged 键不带 tag、
dump 样本条目含重复导致「去重更干净」是错觉）。
