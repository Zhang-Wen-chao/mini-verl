# RLHF 阅读笔记

**公开来源:** [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)(Christiano et al., 2017) → [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)(Ouyang et al., 2022, InstructGPT)。

## 核心观点

**用人类偏好训练奖励模型，再用 RL 优化它。** 人类打分太贵太慢，所以先让人类对回答做**成对比较**，训练一个自动打分器（reward model），然后用 PPO 让模型朝着"高分回答"优化。这就是对齐（alignment）的经典三步走。

## 重要创新

- **奖励建模（reward modeling）**:人类比较 → Bradley-Terry 模型学打分，把人类偏好编码成可优化的信号。
- **三步走管线**:SFT（让模型会说话）→ RM（学打分）→ PPO RL（优化）。
- **KL 约束**:RL 阶段加 `KL(新策略 || 参考模型)` 惩罚，防止模型为了刷分偏离原始能力（reward hacking）。
- **InstructGPT 的缩放验证**:在 1.3B/6B/175B 上证明小 RM 能对齐大模型，GPT-3 → InstructGPT。

## 为什么重要

- ChatGPT 的技术底座，整个"对齐"领域的起点。
- 它暴露的核心问题——reward hacking、KL 平衡——是所有后续 LLM RL 都要处理的。

## 读什么

- 为什么不能直接让人类打分（太贵），而要训 RM。
- PPO 阶段为什么必须配 KL 惩罚。

## mini-verl 的转化

本仓库的 GRPO 与 RLHF 的 PPO 阶段同源，但**去掉了 RM 和 Critic**：规则奖励（`RuleRewardWorker`）替代 RM，组内相对分数替代 Critic。`rl-crash-course.md` §4 有 RLHF 三步走的详解。
