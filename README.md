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

## 从这里开始

| 如果你想知道 | 先读 | 你会得到什么 |
|---|---|---|
| **项目现在在哪、下一步做什么？** | [当前状态与路线图](docs/project-status.md) | 已完成、未做、下一步的唯一权威入口 |
| **4B 实验究竟取得了什么结果？** | [Qwen3.5-4B / 679-step 结果](docs/results/qwen3.5-4b-grpo-679-step.md) | 训练配置、评测协议、3.4× held-out 结果和边界 |
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

## 当前计划

1. 固化现有 679-step 的模型、数据、prompt、评分和 held-out 协议，作为后续对照基线。
2. 在不改变上述契约的前提下，进行 formal-10k 的单变量数据规模对照。
3. 把训练后期答案变长、奖励设计和评分器鲁棒性作为下一轮显式变量。
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
