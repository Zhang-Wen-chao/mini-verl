# 文档导航

不要从文件名猜文档的重要性。按你的问题从下面进入；每类文档只承担一种职责。

## 1. 当前决策：项目现在在哪、接下来做什么

- [项目状态与路线图](project-status.md)：**唯一的当前状态入口**。先看这里。

## 2. 已取得的结果：实验到底证明了什么

- [Qwen3.5-4B / 679-step GRPO 结果](results/qwen3.5-4b-grpo-679-step.md)：正式实验、验收、held-out 结果。
- [Qwen3.5-4B / 170-step GRPO 开发对照](results/qwen3.5-4b-grpo-170-step-development-ablation.md)：no-std 与 standard GRPO 的单 seed 受控筛选；明确不是 held-out 算法结论。
- [4B PPO/GAE 可行性记录](../official_verl/docs/runlogs/2026-08-22-qwen3.5-4b-ppo-gae-feasibility-plan.md)：真实 Critic 校准的固定契约、验收条件与当前 runtime 阻断。
- [510 → 679 训练中回落分析](results/step-510-to-679-regression-analysis.md)：为什么监控分数表面回落，以及评分器误杀边界。

## 3. 实现：代码解决了哪些问题

- [mini_verl 架构与实现进度](architecture/mini-verl-architecture.md)：trajectory、rollout、reward、GRPO、同步、性能实验。
- [mini_verl 性能实验](engineering/mini-verl-performance.md)：受控 benchmark 的原始口径、收益与负结果。

## 4. 操作：如何运行与复现

- [运行指南](guides/runbook.md)：测试、toy/HF benchmark、GPU pipeline。
- [L20 实验与评测经验](operations/l20-lessons-learned.md)：解码超时、评分器、容器和 GPU 诊断。
- [官方 verl 实验索引](../official_verl/README.md)：官方训练栈的脚本、固定输入与历史证据。

## 5. 学习：理解术语与算法

- [RL / PPO / GRPO / DPO 速成](guides/rl-crash-course.md)。
- [极小 PPO actor--critic 与 GRPO 对照](guides/ppo-grpo-toy-comparison.md)：同一 toy 环境中，哪些测试可共享、Critic/GAE 与组内 advantage 如何不同，以及昨晚 170-step 对照实际测了什么。

## 阅读原则

`official_verl/docs/history/` 和 `official_verl/docs/runlogs/` 是**归档证据**，不是当前计划。
要知道下一步，只看 [项目状态与路线图](project-status.md)。
