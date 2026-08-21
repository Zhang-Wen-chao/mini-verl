# 项目状态：mini-verl / official-verl-grpo

更新：2026-08-21。分支：`official-verl-grpo`。

## 一句话结论

这个分支已经不只是“计划做一个 GRPO 框架”：它同时完成了一个可验证的最小 GRPO
实现，以及一次官方 `verl` 上的 Qwen3.5-4B 真实训练。最强证据是独立 MATH held-out
200 题从 **5/200（2.5%）** 到 **17/200（8.5%）**，即 **3.4×**。

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
```

| 层级 | 状态 | 做成了什么 | 应该读什么 |
|---|---|---|---|
| 最小框架 `mini_verl/` | 已完成核心闭环 | trajectory 契约、GRPO + KL、HF rollout/trainer、策略同步、checkpoint、DDP smoke、长度调度与性能观测 | [README：框架进度](README.md#mini-verl-实现进度框架层)、[RUNBOOK](RUNBOOK.md) |
| 官方训练系统 `official_verl/` | 已完成 | 锁定官方 `verl`、FSDP2 + vLLM、preflight、数据转换、规则奖励、checkpoint 与 clean exit | [official_verl/README](official_verl/README.md)、[0.6B smoke](official_verl/SMOKE_RESULT.md) |
| 4B 训练质量 | 已完成第一条有效实验 | Qwen3.5-4B、2037 训练题、679 step、4×L20、独立 held-out 正向提升 | [679-step 验收](run_679_acceptance.md) |
| 评测与实验可靠性 | 已识别并修复关键问题 | 训练/评测去重、答案格式归一化、逐题落盘、单题超时、评测回落诊断 | [回落分析](analysis_regression_510_vs_679.md)、[经验记录](lessons-learned.md) |

## 当前最值得展示的亮点

### 1. 不是只跑通，而是有 held-out 质量证据

| 实验 | 训练 | 评测 | 结果 | 结论 |
|---|---|---|---|---|
| Qwen3.5-4B GRPO 正式 run | OpenR1-Math 过滤后 2037 题；679 step；4×L20 | MATH-lighteval test 随机 200 题；训练未见；base/final 同 prompt、greedy 与归一化评分 | base 5/200（2.5%）→ 679 17/200（8.5%） | **3.4×**，可作为“该训练设置产生 held-out 改善”的证据 |
| 训练中监控 | 同上 | 64 题、训练集零重叠 | 42/64 → 最高 56/64；最终原始评分 53/64，人工确认约 55/64 | 验证训练中存在信号；最终结论仍以独立 200 题为主 |

完整配置、checkpoint、评测节点、训练信号和边界见 [run_679_acceptance.md](run_679_acceptance.md)。

### 2. 对失败和负结果也留了证据

- 0.6B / 16-step 官方 smoke 被明确标为**系统成功、质量结论拒绝**，没有把它包装成提升。
- 2 trainer + 2 rollout 的 4B 拓扑第二次 actor update OOM；通过 3 trainer + 1 rollout
  的验证解决了该资源边界，而不是无依据地继续扩训。
- 510 → 679 的表面回落被逐题核查，确认其中两题是分数/小数等价答案被字符串评分器误杀。
- 变长批处理和 prefetch 的性能结论同时记录收益与代价，例如 token budget 可能降低吞吐，
  双卡 prefetch 在生成主导 workload 中只能隐藏有限 rollout 时间。

### 3. mini 框架的价值在“能验证和定位”，不是重复造完整 verl

`mini_verl/` 刻意不复制 Ray、多机编排或生产级服务系统。它把下列边界拆开并加以测试：

```text
rollout → trajectory / old logprob → reward → group advantage
        → GRPO + reference KL → optimizer update → policy-version sync
```

这使得算法正确性、策略陈旧性、padding、长度准入、checkpoint 和 rollout/train
重叠可以分别验证，而不是都隐藏在一次大规模训练里。

## 目录导航

```text
README.md                         # 首页：成果、入口、框架进度
PROJECT_STATUS.md                 # 当前状态、亮点、边界、下一步（本文件）
run_679_acceptance.md             # 4B 正式 run 的核心验收证据
analysis_regression_510_vs_679.md # 训练中评测回落的逐题诊断
lessons-learned.md                # 真实训练与评测踩坑
rl-crash-course.md                # PPO / GRPO / DPO 等算法直觉
RUNBOOK.md                        # 本地与 GPU 验证命令
PERFORMANCE_REPORT.md             # mini 框架的性能实验原始解释
mini_verl/                        # 最小 GRPO 实现
tests/                            # 契约、正确性、集成与脚本测试
benchmarks/                       # toy / HF / pipeline 性能对照
official_verl/                    # 官方 verl 的可复现实验契约、脚本与运行记录
```

## 当前边界：哪些还没有做

| 项目 | 状态 | 原因 / 位置 |
|---|---|---|
| PPO actor-critic / GAE | 未做 | `mini_verl` 当前专注 GRPO；PPO 需要 value/critic 的额外资源与稳定性设计 |
| 学习型 Reward Model、DPO、KTO | 未做 | 当前真实实验使用可审计的数学规则 reward，避免把奖励模型误差与 GRPO 混在首条质量结论中 |
| vLLM / SGLang 作为 `mini_verl` rollout backend | 未做 | 最小框架已有 HF backend；官方验证已使用 vLLM，下一步再决定是否抽象回接 |
| 多机、异构硬件、生产级容错 | 未做 | 有意不纳入教学/验证型最小实现 |
| formal-10k 对照实验 | 待做 | 先固化当前 4B 基线、评分与长度治理，再扩大数据规模 |

## 下一步

1. **固化 679-step 基线。** 将模型、数据子集、prompt、归一化评分、seed、4-GPU 拓扑和
   held-out 协议作为之后所有对照的固定基准。
2. **做 formal-10k 的单变量对照。** 保持模型、奖励、评测和资源拓扑不变，只扩大训练数据，
   并报告 held-out、reward 分布、KL、长度、截断比例、step time 与 checkpoint。
3. **先治理长度与 reward。** 训练后期 response length 接近 cap，而 reward 不惩罚冗长；
   新实验应把长度目标和评分鲁棒性作为显式设计变量，而不是把分数变化直接归因给数据规模。
4. **回写最小实现。** 只把已在官方链路中验证的 `dataset → rollout → verifier reward →
   group advantage → logprob → update → metrics` 语义映射回 `mini_verl`，不复制完整生产编排。

## 验证状态

2026-08-21 在当前工作区运行 `python3 -m unittest discover -s tests -v`：**106 项通过，
28 项跳过**。跳过项依赖本机未安装的可选 PyTorch/Transformers；纯 Python 的协议、
GRPO reference、控制器、奖励、官方脚本与数据契约测试均已执行。
