# 强化学习动画视频笔记（B站「耿直哥」系列）

> 本目录整理了 B 站「耿直哥」强化学习系列视频的中文字幕，转写成结构化的学习笔记。
> 所有笔记都：去掉了时间戳、修正了语音识别错别字（如 "clip 函数"、"信赖域"、"ε" 等）、按主题分节。
> 原始视频（mp4 + 六语种 .ass 字幕）在本地 `~/Downloads/性感的【强化学习】/` 目录。

## 推荐阅读顺序

建议按算法进化史阅读：**TRPO → PPO → GRPO → DAPO**（优化主线），再看 RLHF/DPO（对齐支线）。

| 笔记 | 对应视频 | 主题 |
|---|---|---|
| [01 · 近十年 RL 主流模型综述](01-rl-mainstream-algorithms-overview.md) | 全网最"强"系列：近十年强化学习主流模型一网打尽 | 系列导读：为什么学 RL、怎么学 |
| [02 · TRPO 算法动画拆解](02-trpo-animation-explained.md) | 动画手撕神级 TRPO 算法 | 信赖域（trust region）、KL 约束、重要性采样 |
| [03 · PPO 算法动画拆解](03-ppo-animation-explained.md) | PPO算法：强化学习头牌 | clip 函数、动态信赖域、PPO 思想谱系 |
| [04 · GRPO 算法动画拆解](04-grpo-animation-explained.md) | 大模型强化学习：GRPO算法 | 去价值网络、组内相对优势、solution-level |
| [05 · DAPO 算法动画拆解](05-dapo-animation-explained.md) | DAPO算法：稳重进化 | clip 改进、输出约束、token 级归一化、长度感知奖励 |
| [06 · RLHF/DPO 爬山讲透](06-rlhf-dpo-mountain-analogy.md) | 带女朋友爬山讲明白 RLHF 和 DPO | RLHF 三步走、几个网络、DPO 闭式解 |

## 快速对照：TRPO / PPO / GRPO / DAPO 进化一览

| | TRPO | PPO | GRPO | DAPO |
|---|---|---|---|---|
| 核心机制 | 信赖域（KL 约束） | clip 函数（动态信赖域） | 组内相对优势（去 Critic） | clip 改进 + 4 项稳定技巧 |
| 价值网络 | 需要 | 需要（Critic） | **不需要** | 不需要 |
| 优势计算 | GAE + Critic | GAE + Critic | 组内归一化 | 组内归一化（token 级） |
| 粒度 | step | step（token 级） | solution（序列级） | solution + token 混合 |
| 适合场景 | 通用 RL | 通用 RL（含密集奖励控制） | LLM 稀疏奖励（数学/代码） | LLM 长序列推理 |
| 一句话 | 谨慎的老太太 | 精明的小脚女人 | 勇敢的野小子 | 成熟的青年 |

> 人物比喻出自 DAPO 视频（TRPO=谨慎老太太 / PPO=精明小脚女人 / GRPO=勇敢野小子 / DAPO=成熟青年）。
