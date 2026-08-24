# DQN 阅读笔记

**公开来源:** [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)(Mnih et al., 2015, Nature)。

## 核心观点

把 Q-learning 和深度网络结合，**让神经网络直接从高维像素学玩 Atari 游戏**。价值派（value-based）的代表：不直接学策略，而是学"每个动作值多少分"，策略就是每步选 Q 值最大的动作。

## 重要创新

- **经验回放（experience replay）**:把经历存进 buffer，随机抽样训练，打破样本间的时间相关性。
- **目标网络（target network）**:用一份"冻结"的旧网络算 Q 目标，防止更新目标随参数一起漂移导致发散。
- **端到端从像素学习**:CNN 直接吃游戏画面，是深度 RL 的开山之作。

## 为什么重要

- 证明了"深度学习 + 强化学习"能解决高维感知任务，是 AlphaGo 的前置技术。
- 它奠基的**回放 + 目标网络**两个技巧，后来被几乎所有深度 RL 算法复用。

## 读什么

- Q-learning 为什么容易发散，回放和目标网络各解决什么。
- value-based 与 policy-based（REINFORCE 那条线）的思维差异。

## mini-verl 的转化

本仓库聚焦 LLM RL（policy-based 线），不实现 DQN。但"经验回放"思想与 π₀.₆* 的经验回放实验、以及 GRPO 的 batch 采样逻辑有概念对应——理解它有助于区分"用旧数据再训练"和"在线采样"。
