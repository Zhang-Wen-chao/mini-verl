# DDPG 阅读笔记

**公开来源:** [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)(Lillicrap et al., 2016)。

## 核心观点

把 DQN 的思路扩展到**连续动作空间**。DQN 靠枚举所有动作选最大的 Q，但连续动作（如机器人关节力矩）无法枚举——DDPG 用一个"演员网络"直接输出最优动作，配合"评论家网络"打分。**Actor-Critic 架构的经典落地。**

## 重要创新

- **确定性策略梯度（DPG）**:演员输出确定性的动作（不是概率分布），配合 Q 值梯度更新。
- **Actor-Critic 双网络**:演员（actor）决定动作，评论家（critic）评估动作，互相促进。
- **复用 DQN 技巧**:经验回放 + 目标网络（软更新，slowly copy）都搬了过来。

## 与 PPO 的对比

| | DDPG | PPO |
|---|---|---|
| 动作 | 连续（确定性输出） | 离散/连续（概率分布采样） |
| 探索 | 靠给动作加噪声 | 靠策略本身的随机性 |
| 价值网络 | 需要（critic） | 需要（critic） |
| 训练 | off-policy（用旧数据） | on-policy（用最新数据） |

## 读什么

- 为什么 DQN 不能直接用连续动作，DDPG 怎么解决。
- actor-critic 架构如何把"学动作"和"学评价"分离。

## mini-verl 的转化

PPO/GRPO 里的"actor"一词正来自 actor-critic 传统——被训练的模型就是 actor，GRPO 只是把 critic 换成了组内相对分数。理解 DDPG 能帮你回答"为什么叫 actor 更新"。
