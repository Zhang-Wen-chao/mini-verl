# DPO 阅读笔记

**公开来源:** [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)(Rafailov et al., 2023)。

## 核心观点

**RLHF 的 PPO 阶段其实可以闭式解掉。** 与其训 RM 再跑 PPO，不如直接用偏好数据（"回答 A 比 B 好"）优化语言模型——目标是让"好的回答概率高、差的回答概率低"，一步到位。论文标题的意思就是：**你的语言模型本身就可以当奖励模型用。**

## 重要创新

- **闭式解**:把"RM + PPO RL"两步压缩成单个损失函数，不需要训练 RM、不需要 RL 循环。
- **DPO 损失**:基于新旧策略概率比的偏好对数似然，带参考模型做隐式 KL 正则。
- **省资源**:省掉 RM 训练和在线采样，训练轻量、稳定、可复现。

## 代价

- **没有探索**:纯离线优化，模型不会在训练中尝试新行为——上限低于好的 RL。
- **对偏好数据质量敏感**:依赖"哪个回答更好"标注的准确性。

## 读什么

- DPO 怎么把 RLHF 的两步推导成一个闭式目标（核心是"策略即奖励模型"的洞察）。
- 与 PPO/GRPO 的对比：离线偏好优化 vs 在线 RL。

## mini-verl 的转化

`mini_verl` 现已实现在线版 DPO 闭环：`preference.py` 在每个 prompt 组内用规则 reward 取
最高/最低分构造 chosen/rejected 偏好对，`algorithms/dpo.py` 提供序列级
`-log σ(β·margin)` 目标的 reference/torch 双实现，`HuggingFaceDpoTrainerWorker` 复用
GRPO 的 rollout/reward 阶段做更新，Controller 零改动。这是"在线 DPO"的配对来源
（偏好对来自采样+打分，不是人工标注），更新本身没有 advantage、critic 和 ratio clip；
离线偏好数据集加载与 KTO 仍未做。

面试被问"GRPO 和 DPO 区别"时答：**DPO 可以连 RL 循环都不要，直接在偏好对上一步优化；
GRPO 在 RL 框架里在线采样、组内相对比较。** mini-verl 的 DPO 借用了 RL 循环的采样来
构造偏好对，但优化目标就是纯 DPO。（面试详稿 §7 有退路话术。）
