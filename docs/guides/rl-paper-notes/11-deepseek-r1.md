# DeepSeek-R1 阅读笔记

**公开来源:** [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)(DeepSeek-AI, 2025)。

## 核心观点

**纯 RL 就能让模型长出长推理能力。** 在数学/代码任务上，只用 RL（GRPO + 规则奖励）训练，不靠大量人工标注的思维链，模型自己涌现出"反思、验证、长链推理"——这是 LLM RL 从"对齐"走向"能力提升"的里程碑。

## 重要创新

- **R1-Zero:纯 RL 无 SFT**:直接在 base 模型上跑 RL，验证了"推理能力可以纯 RL 涌现"，但输出可读性差。
- **两阶段训练**:R1-Zero 后加少量冷启动 SFT 再 RL，兼顾能力与可读性 → R1。
- **规则奖励（rule-based reward）**:数学用答案校验、代码用编译/测试，不用学 RM——可扩展、不钻空子。
- **大规模蒸馏**:R1 的能力蒸馏到 1.5B–70B 小模型，开源生态受益。
- **"aha moment"**:训练中模型自发出现反思行为（如"等等，我再验算一下"），无显式引导。

## 为什么重要

- 证明"推理 RL"独立于"对齐 RL"，是 RL 在 LLM 上的第二次爆发。
- GRPO 因此成为大模型推理 RL 的事实标准，被 verl 等框架广泛采用。

## 读什么

- R1-Zero vs R1:纯 RL 和"冷启动 SFT + RL"的取舍。
- 规则奖励 vs 学习型 RM:为什么前者更适合可验证任务。

## mini-verl 的转化

本仓库 679 步 GRPO 训练正是 R1 路线的缩小版（Qwen3.5-4B + 规则奖励 + GRPO）。`rl-crash-course.md` 用数学题例子讲解的"GRPO 训练在干嘛"就是 R1 的机制。
