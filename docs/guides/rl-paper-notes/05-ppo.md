# PPO 阅读笔记

**公开来源:** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)(Schulman et al., 2017)。

## 核心观点

TRPO 信赖域的**廉价版**。不显式解 KL 约束，而是用一行 `clip` 函数把"新旧策略概率比"限制在 `[1-ε, 1+ε]` 内——更新太猛的部分直接截断。简单、稳定、工业标准。

## 重要创新

- **clip 目标函数**:`min(ratio·A, clip(ratio)·A)`，ratio 是新旧概率比，A 是 advantage。
  - A>0（比预期好）：鼓励但限制放大倍数；
  - A<0（比预期差）：惩罚但限制打压幅度。
- **GAE（广义优势估计）**:用 λ 在"偏差小"和"方差小"之间折中，累计多步 advantage。
- **Critic 价值网络**:估计状态价值，算 advantage = 实际 − 预期，压方差。

## 为什么是工业标准

- 实现简单（一个 clip 函数），调参友好，不需要 TRPO 的昂贵优化。
- 适用离散/连续、游戏/机器人/语言，通用性最强。
- 大模型 RL 的 RLHF 阶段最初就用 PPO。

## 读什么

- clip 对正负 advantage 的不对称行为（为什么"只限制往坏的方向"）。
- GAE 的 γ 和 λ 各管什么。

## mini-verl 的转化

`mini_verl/` 有完整 PPO loss 实现与单测（`rl-crash-course.md` 有爬山比喻详解）。面试详稿 5b 节有 clip 对正负 advantage 不对称行为的进阶问答。
