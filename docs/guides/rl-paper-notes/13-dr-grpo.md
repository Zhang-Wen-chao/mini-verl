# Dr. GRPO 阅读笔记

**公开来源:** [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)(Li et al., 2025, 即 Dr. GRPO)。

## 核心观点

**GRPO 在 R1-Zero 式训练里有一个被忽视的陷阱：对同组内、长度不同的回答，token 级概率乘积会让"较长的回答"在损失里被不公平地加权。** 论文先"诊断"GRPO 何时失效（Dr. 的含义），再给出修正——把 GRPO 损失改造成**逐 token 的优势加权**，不再按整句归一化。

## 重要创新

- **诊断:长度偏差**:`Π p(token)` 使长回答的联合概率天然更小，组内比较时长回答吃亏（即使每 token 概率更高）。长回答被系统性低估。
- **修正:逐 token 归一化**:不再对整句算一个 advantage，而是每个 token 单独算 `(p_token − 基线) / 标准差`，消除长度混淆。
- **理论分析**:证明原版 GRPO 的梯度隐含一个"长度惩罚项"，这是性能次优的根因。
- **实验验证**:修正版在 R1-Zero 类设置上持续超过原版。

## 为什么重要

- 是"理解 GRPO 为什么 work/不 work"最透彻的一篇，直接指出公式层面的缺陷。
- DAPO 的 token 级归一化与 Dr. GRPO 的修正同向，说明这个洞察是共识。

## 读什么

- 为什么 `Π p(token)` 让长回答吃亏（长度与联合概率的数学关系）。
- 逐 token vs 整句 advantage 的差异。

## mini-verl 的转化

本仓库 GRPO 用整句级 advantage（标准版）。面试进阶可讲："我知道 GRPO 有个长度偏差陷阱，Dr. GRPO 证明它对长回答不公，DAPO 用 token 级归一化修它"——体现你读过原理而不只是会用。
