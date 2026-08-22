# mini-verl：LLM GRPO 后训练的可验证实验仓库

这个仓库有一条主线：先在 `mini_verl/` 中把 GRPO 的算法和系统边界做成可测试的最小
实现，再在锁定版本的官方 `verl` 上完成真实 Qwen3.5-4B 训练验证。

## 先看结论

训练过程：在去泄漏的 OpenR1-Math **2037 道训练题**上进行 **679 次 GRPO 更新**。
同一批训练集之外的 MATH held-out 200 题只用于评测、不参与任何更新：**未做 GRPO
强化训练的原始 Qwen3.5-4B base checkpoint** 得到 **5/200（2.5%）**；**GRPO 训练
679 step 后的 checkpoint** 得到 **17/200（8.5%）**。即 **绝对 +6.0 个百分点**、
相对 **3.4×**。

### 两套评测，两个都该看

| 评测 | 对比结果 | 提升 | 该怎样理解 |
|---|---|---:|---|
| 训练中固定监控集，64 题 | base 42/64（65.6%）→ **step 510: 56/64（87.5%）** | **峰值 +21.9pp** | 训练过程的强信号；与训练集零重叠，但训练中反复评测，不作为最终泛化结论。最终 step 679 原始评分为 53/64（+17.2pp）；经两题等价答案格式误杀复核后约 55/64（约 +20.3pp）。 |
| 最终 MATH held-out，200 题 | 未做 GRPO base 5/200（2.5%）→ 679-step 17/200（8.5%） | **+6.0pp** | 最终泛化结论：训练未见，只在 base 与最终 checkpoint 各评一次。 |

这是一条受控实验下的正向质量证据，不代表所有数学任务或模型都会得到同样结果；
完整协议、checkpoint、评分器边界与反例都保存在结果文档中。

### 最新进展：PPO 与 GRPO 已完成同样本预算的 5-step 可行性对照

2026-08-22 在同一 Qwen3.5-4B base、相同 2,037 条训练数据、seed、3 trainer + 1 vLLM
拓扑、每步 3 prompts × 4 rollouts = 12 trajectories 的契约下，真实 PPO/GAE + 4B Critic
与 standard GRPO 均完成 5 个 update、最终 checkpoint、rollout 样本及 clean watcher exit。PPO
在保留完整样本预算的前提下需启用 Critic activation offload 才能越过 Critic backward 的显存
峰值；完成后 PPO 训练 loop 用时 8m35s、最后一步 Actor 最大 allocated GPU memory 35.70 GB，
GRPO 分别为 6m42s、31.53 GB。

固定 64 题开发 monitor 的变化是 PPO 41/64 → 45/64，GRPO 40/64 → 42/64。**这不是
算法胜负结论**：只有 5 step、单 seed、两边起点不同，且 64 题已被反复读取。本轮可靠结论是
“真实 4B actor--critic PPO 在当前 4×L20 与完整 12-trajectory 契约下可运行，但比这次 GRPO
短运行消耗更多资源”。完整的契约、首次 OOM 边界、artifact 和后续要求见
[PPO/GRPO 5-step 公平开发对照](official_verl/docs/runlogs/2026-08-22-qwen3.5-4b-ppo-grpo-fair-development-comparison.md)。

### 此前进展：完成 GRPO advantage 标准化的受控开发对照

2026-08-22 还完成了一组新的 4×L20、3 trainer + 1 vLLM rollout 的串行实验：先用
20 step no-std GRPO 校准环境，再从 base 分别运行两个 170-step development run。两段
长实验都完整落盘（170 rollout、`global_step_170`、训练与 watcher exit code 均为 0），
没有 OOM、NCCL 或非有限数值错误。唯一的目标变量是
`algorithm.norm_adv_by_std_in_grpo`：

| 开发 run | 固定 64 题 monitor：step 0 → 170 | 该怎么读 |
|---|---:|---|
| no-std GRPO（不除组内标准差） | 42/64（65.6%）→ 47/64（73.4%） | 稳定完成；mixed-reward groups 均有非零梯度。 |
| standard GRPO（默认除标准差） | 41/64（64.1%）→ 48/64（75.0%） | 同样稳定完成；本次 monitor 增幅略高。 |

这**不是算法胜负结论**：它是单 seed、反复使用的 64 题开发监控，而且两个 run 的初始
monitor 本身不同。MATH held-out 200 在这次筛选中没有被读取。完整的实验契约、
telemetry、限制和下一步见[170-step GRPO 开发对照](docs/results/qwen3.5-4b-grpo-170-step-development-ablation.md)。

## 从这里开始

| 如果你想知道 | 先读 | 你会得到什么 |
|---|---|---|
| **项目现在在哪、下一步做什么？** | [当前状态与路线图](docs/project-status.md) | 已完成、未做、下一步的唯一权威入口 |
| **4B 实验究竟取得了什么结果？** | [Qwen3.5-4B / 679-step 结果](docs/results/qwen3.5-4b-grpo-679-step.md) | 训练配置、评测协议、3.4× held-out 结果和边界 |
| **PPO 与 GRPO 的最新对照说明什么？** | [PPO/GRPO 5-step 公平开发对照](official_verl/docs/runlogs/2026-08-22-qwen3.5-4b-ppo-grpo-fair-development-comparison.md) | 相同 12-trajectory 预算下的可行性、资源代价与严格边界 |
| **此前 GRPO 算法对照说明什么？** | [170-step 开发对照](docs/results/qwen3.5-4b-grpo-170-step-development-ablation.md) | no-std vs standard 的完成证据、telemetry 与不可过度解释的边界 |
| **代码实现了什么？** | [mini_verl 架构与实现进度](docs/architecture/mini-verl-architecture.md) | 数据流、模块职责、已完成的框架能力 |
| **怎样跑测试、benchmark 或实验？** | [运行指南](docs/guides/runbook.md) | 本地、GPU 与官方 verl 的执行命令 |
| **异步训练、PPO、GRPO 是什么？** | [RL 速成课](docs/guides/rl-crash-course.md) | 算法直觉与本仓库的对应关系 |
| **官方 verl 脚本和历史证据在哪？** | [official_verl 实验索引](official_verl/README.md) | 可执行脚本、固定版本、结果、runlog 归档 |

如果只读三份文档，请按这个顺序：

```text
docs/project-status.md
    → docs/results/qwen3.5-4b-grpo-679-step.md
    → docs/architecture/mini-verl-architecture.md
```

## 三分钟跑通最小实现

`mini_verl/` 是独立于官方 `verl` 实验栈的教学实现：不下载模型、可在 CPU 上运行，保留
`rollout → old logprob → rule reward → group advantage → clipped update → policy version`
的数据流。

```bash
python -m pip install -e '.[torch]'
python -m mini_verl.toy
PYTHONPATH=. python examples/toy_ppo_train.py
```

第一条是最小 GRPO 闭环，第二条是同一 categorical 环境中的 PPO actor--critic/GAE 教学对照；
它们用于理解和单测，不等同于真实 4B 训练。命令、验收条件和 GPU benchmark 见
[运行指南](docs/guides/runbook.md)，实现边界见
[mini_verl 架构与实现进度](docs/architecture/mini-verl-architecture.md)。

## 当前计划

1. 固化 679-step held-out 基线、170-step GRPO 对照及 PPO/GRPO 5-step artifact；不把短开发遥测写成算法胜负。
2. 固定新的 development / final 划分、停止规则与 seed 数；不复用 64 题或 MATH held-out 200 来选择算法。
3. 资源允许时，先以该冻结契约进行 20-step PPO/GRPO development 对照，再决定多 seed 或更长 run。
4. 将官方路径中已验证的语义继续回写到 `mini_verl/`，不复制 Ray/FSDP/vLLM 的完整生产编排。

详见 [当前状态与路线图](docs/project-status.md#下一步)。

## 目录职责

```text
README.md                 # 唯一首页：结论、阅读顺序、当前计划
docs/                     # 人读文档，按状态 / 结果 / 架构 / 指南 / 运维分类
mini_verl/                # 最小、可测试的 GRPO 实现
tests/                    # 正确性、契约和集成测试
benchmarks/               # toy、HF、pipeline 性能对照
examples/                 # 最小可运行示例
official_verl/            # 官方 verl 的可执行脚本、预检和实验归档索引
```

文档总索引见 [docs/README.md](docs/README.md)。
