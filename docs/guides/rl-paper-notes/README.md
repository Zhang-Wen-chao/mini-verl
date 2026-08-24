# RL 论文阅读笔记

> 面向"快速了解强化学习全部内容"的论文速读笔记。每篇只讲**核心观点 + 重要创新**，
> 不深入公式推导——目标是建立整个 RL 谱系的地图，需要细节时再读原文。
> 格式对齐 mini-pi 的 `paper-notes`：公开来源 → 核心观点 → 重要创新 → 读什么。

## 怎么读（两条主线）

RL 论文可以沿两条主线读，交叉点在大模型 RL：

1. **经典 RL 主线**（value-based → policy-based → actor-critic）：DQN → DDPG → TRPO → PPO → SAC，理解"怎么更新策略"的演进。
2. **LLM RL 主线**（从人类反馈到推理）：RLHF → DPO → GRPO → DeepSeek-R1 → DAPO，理解"大模型怎么用 RL 对齐和提升推理"。

建议先读 [README 导航](../README.md) 和 [RL 速成](../rl-crash-course.md) 建立术语，再按下面顺序过一遍。

## 论文清单（按阅读顺序）

### 第一组：经典 RL 基础（怎么更新策略）

| # | 论文 | 一句话 | 笔记 |
|---|---|---|---|
| 1 | REINFORCE（Williams 1992） | 最朴素的策略梯度：用整段回报的正负推概率 | [01-reinforce.md](01-reinforce.md) |
| 2 | DQN（Mnih 2015） | 用深度网络 + 经验回放 + 目标网络学 Q 值 | [02-dqn.md](02-dqn.md) |
| 3 | TRPO（Schulman 2015） | 信赖域约束：更新不许偏离旧策略太远 | [03-trpo.md](03-trpo.md) |
| 4 | DDPG（Lillicrap 2016） | 连续动作的 actor-critic：确定性策略 + 回放 | [04-ddpg.md](04-ddpg.md) |
| 5 | PPO（Schulman 2017） | TRPO 的廉价版：一行 clip 代替信赖域 | [05-ppo.md](05-ppo.md) |
| 6 | SAC（Haarnoja 2018） | 最大熵：在"拿高分"和"保持随机探索"间平衡 | [06-sac.md](06-sac.md) |

### 第二组：从人类反馈到偏好（对齐支线）

| # | 论文 | 一句话 | 笔记 |
|---|---|---|---|
| 7 | RLHF（Christiano/Ouyang 2017/2022） | 人类偏好 → 奖励模型 → PPO 对齐 | [07-rlhf.md](07-rlhf.md) |
| 8 | DPO（Rafailov 2023） | 跳过奖励模型，直接用偏好对优化 | [08-dpo.md](08-dpo.md) |
| 9 | KTO（Ethayarajh 2023） | 只用"好不好"（单样本），不用成对比较 | [09-kto.md](09-kto.md) |

### 第三组：LLM 推理 RL（前沿主线）

| # | 论文 | 一句话 | 笔记 |
|---|---|---|---|
| 10 | GRPO（DeepSeekMath 2024） | 去 Critic，组内相对优势 | [10-grpo.md](10-grpo.md) |
| 11 | DeepSeek-R1（2025） | 大规模 RL 让模型学会长推理 | [11-deepseek-r1.md](11-deepseek-r1.md) |
| 12 | DAPO（2025） | clip 改进 + 4 项稳定大模型 RL 的技巧 | [12-dapo.md](12-dapo.md) |
| 13 | Dr. GRPO（2025） | 理解 GRPO 何时失效，并给出改进 | [13-dr-grpo.md](13-dr-grpo.md) |

## mini-verl 的对应

每篇笔记末尾的"mini-verl 的转化"说明该算法在本仓库的实现位置或实验对应：

- **PPO / GRPO**：`mini_verl/` 有完整的 loss 实现与单测（[架构文档](../../architecture/mini-verl-architecture.md)）
- **GRPO 实战**：`official_verl/` 的 Qwen3.5-4B 679 步训练与 0.6B 优化对照
- **KL 约束**：`rl-crash-course.md` §6 记录了 KL 系数的调参直觉
- 其余论文多为概念理解，未实现（本仓库聚焦 LLM RL，不是通用 RL 框架）
