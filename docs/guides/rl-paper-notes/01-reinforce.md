# REINFORCE 阅读笔记

**公开来源:** [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://paperswithcode.com/method/reinforce)(Williams, 1992)。

## 核心观点

最朴素的策略梯度算法。**"做过的动作里，分数高的多来点，分数低的少来点。"** 不学价值函数，直接让策略根据整段轨迹的回报调整动作概率——回报高的动作概率上调，低的概率下调。

## 重要创新

- **策略梯度定理的落地**:用 `∇log π(a|s) × 回报` 作为更新方向，把"调概率"和"回报"直接挂钩。
- **不需要价值网络**:整个算法只有一个策略网络，概念最简单。
- **蒙特卡洛采样**:用完整轨迹的累计回报估计梯度（无偏但有高方差）。

## 关键局限

- **方差极大**:同一状态两次采样回报可能一正一负，梯度方向相反，学得慢。
- **无 credit assignment**:整段轨迹一个总回报，分不清哪个动作该负责。

## 读什么

- 为什么策略梯度可以用"回报 × 概率梯度"作为更新方向。
- 高方差从哪来，后续算法（baseline、critic）都是为压方差。

## mini-verl 的转化

GRPO 的 advantage 本质上就是"带 baseline 的策略梯度"——用组内平均回报当 baseline 压方差。REINFORCE 是理解 GRPO advantage 公式的起点。
