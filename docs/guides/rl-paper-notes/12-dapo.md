# DAPO 阅读笔记

**公开来源:** [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)(ByteDance, 2025)。

## 核心观点

**大模型 RL 在规模上会踩 4 个坑，DAPO 逐个修。** 在 8K 卡级规模复现 DeepSeek-R1 时发现：熵崩塌、奖励噪声、训练不稳定、长度膨胀——每修一个就提一点，合起来才让大规模 RL 真正可复现。

## 重要创新（四个关键技巧）

- **Clip-Higher**:只限制 ratio 的下界（防崩塌），不设上界——让高 advantage 样本充分学习，防止熵崩塌。
- **动态采样（dynamic sampling）**:过滤掉全对/全错的 prompt，只保留"组内有区分度"的样本，稳定训练。
- **Token 级 loss 归一化**:把 loss 除以非 padding 的 token 数（而非序列数），长回答不被过分强调。
- **长度奖励（overlong reward shaping）**:对超长回答给负奖励，抑制"越写越长"的膨胀（你仓库踩过的坑）。

## 为什么重要

- 是 GRPO 之后最重要的 LLM RL 工程论文，公开了 4 个可复现的稳定技巧。
- 直接回应了"为什么我复现 R1 不收敛"——多半是这 4 个坑之一。

## 读什么

- 熵崩塌和 Clip-Higher 的关系（为什么传统 clip 会杀死探索）。
- 长度膨胀的奖励整形做法。

## mini-verl 的转化

你仓库的 679 步实验踩过**长度膨胀**（response 138→270+）——DAPO 的第 4 个技巧就是解药。`rl-crash-course.md` §6 记录了这个问题；`rl-video-notes/05-dapo` 有动画详解。
