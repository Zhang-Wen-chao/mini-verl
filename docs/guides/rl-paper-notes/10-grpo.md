# GRPO 阅读笔记

**公开来源:** [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)(Shao et al., 2024)。

## 核心观点

**把 PPO 的 Critic 扔掉，用"同一道题的多个回答的组内相对分数"算 advantage。** 大模型场景下训 Critic 又贵又难（"这个状态值多少分"对语言太难定义），GRPO 用组内对比天然替代它——**显存省一半，训练更稳。**

## 重要创新

- **组内相对 advantage**:同一 prompt 采样 N 个回答，`advantage = (自己的分 − 组平均) / 组标准差`。
- **去 Critic**:不再需要价值网络，模型规模不翻倍。
- **稀疏奖励的救星**:数学题 reward 是 0/1，全错时组内对比仍能分出"谁更接近对"，信号不消失。
- **沿用 PPO 的 clip + KL**:策略更新仍是概率比 + clip，只是 advantage 来源变了。

## 与 PPO 的关系

不是全新算法，是"给大模型场景裁剪过的 PPO"——换掉的是 advantage 的计算方式，保留的是稳定更新机制。关键区别就两条：去 Critic、回答级（solution-level）而非 token 级。

## 读什么

- advantage = (r − μ)/σ 的组内归一化，以及 σ=0（全对/全错）时的退化处理。
- 为什么回答级 advantage 对语言任务够用。

## mini-verl 的转化

**本仓库核心算法**：`mini_verl/` 有完整 GRPO loss + 组内 advantage 实现与单测；官方 verl 实战（4B 679 步 + 0.6B 优化对照）都用它。`rl-crash-course.md` §3.3 和面试详稿 §4 有完整话术。
