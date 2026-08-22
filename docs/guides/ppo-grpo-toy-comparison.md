# 极小 PPO actor--critic 与 GRPO：同一测试，什么相同、什么不同

这页对应仓库里的两段可运行教学代码：

- `examples/toy_grpo_train.py`：现有 GRPO。
- `examples/toy_ppo_train.py`：新增的极小 PPO actor--critic。

它们**不是**对 Qwen3.5-4B 做 PPO 的实验，也不报告“PPO 比 GRPO 强”的结论。目的只是把
PPO 的 Critic、GAE 和 value loss 与 GRPO 的组内 advantage 变成同一环境下可检查的代码。

## 能不能用一样的测试？

可以共享大部分测试脚手架，但不能把两种 advantage 的数学期望硬说成一样。两者都使用同一个
16-prompt、8-answer 的离散任务；每道题只有一个正确 token，reward 都是 `0/1`，并以相同的
`pass@1`（每题 argmax 是否为正确 token）检查策略是否学会。两者也都保留 rollout 时的
`old_logprob`，再用同一个 PPO-style ratio/clip 控制 policy 不要一次改太远。

```text
共同部分
prompt → sample answer → rule reward → old/new probability ratio + clip → 更新 actor
                                  └──────────── pass@1 ────────────┘

PPO 特有
reward + Critic V(prompt) → GAE advantage、return → actor loss + value loss

GRPO 特有
同一 prompt 的 8 个 reward → group mean/std → group-relative advantage → policy loss
```

| 能共用的东西 | PPO | GRPO |
|---|---|---|
| 环境、题目、0/1 reward、采样数量 | 是 | 是 |
| `old_logprob`、新旧概率比、clip、可选 reference KL | 是 | 是 |
| `pass@1` 的训练前后检查 | 是 | 是 |
| advantage 的数值测试 | **否**：由 Critic/GAE 得来 | **否**：由同题组内 reward 得来 |
| Critic 的 value/return 回归测试 | 有 | 无 |

仓库单测明确验证了“actor 部分可比”：给定相同 `new_logprob`、`old_logprob` 和 advantage，
`ppo_loss_reference(...).actor_loss` 与 `grpo_loss_reference(...).policy_loss` 相等。差异发生在
**这份 advantage 从哪里来**，而不是发生在“提高正确答案概率”的基本 policy update 上。

## Critic 和 GAE 在这个 toy 中做了什么？

PPO 的 Critic 是一个“先猜这题平均能拿多少 reward”的 value function。在这个最小例子中，它是
一个长度为 16 的可训练表：每个 prompt 一个 `V(prompt)`，而不是为了教学再加一整个神经网络。

每条 episode 只生成一个 token 就结束，因此：

```text
advantage = reward - V(prompt)
return    = reward
value loss = 0.5 × (V(prompt) - return)^2
```

代码仍通过通用 GAE 函数计算这件事；在真正多步环境中则是：

```text
delta_t = reward_t + gamma × V(next_state) - V(state)
advantage_t = delta_t + gamma × lambda × 后续 advantage
return_t = advantage_t + V(state)
```

所以，reward 为 `[0, 0, 1, 1]`、Critic 都预测 `0.25` 时，PPO 的 advantage 是
`[-0.25, -0.25, 0.75, 0.75]`：它表示“相对 Critic 预期的意外程度”。同样 reward 在 GRPO
的一组回答中会先减去组平均值，再可选择除以组标准差：它表示“相对同题其他回答的好坏”。

## 昨晚的 170-step 实验到底做了什么？

它**没有训练 PPO，也没有训练 Critic，更不是 PPO vs GRPO**。它只在 GRPO 内部切换一个
变量：`algorithm.norm_adv_by_std_in_grpo`。

| development run | 固定 64 题 monitor：step 0 → 170 | 测到的东西 |
|---|---:|---|
| no-std GRPO | 42/64（65.6%）→ 47/64（73.4%） | 组内 reward 减均值，但不除组内 std |
| standard GRPO | 41/64（64.1%）→ 48/64（75.0%） | 组内 reward 减均值后再除组内 std（默认） |

两段 170-step run 都是从 base 独立开始，固定训练数据、步数、基础配置和 64 题开发监控；先跑
20 step no-std calibration，是为了确认 4×L20 的 rollout/trainer 管线、梯度和 checkpoint 正常，
再投入长 run。结果说明两种设置在这次单 seed、短预算下都能稳定跑完，standard 这一次的
monitor 增幅略高。

它**不能**说明 standard GRPO 更好，原因有三点：两个 run 的初始 monitor 已相差 1 题；只有
一个 seed；同一 64 题被反复读取，因此它是 development monitor，不是独立 final test。MATH
held-out 200 在这轮筛选没有读取。完整 telemetry 与边界见
[170-step GRPO 开发对照](../results/qwen3.5-4b-grpo-170-step-development-ablation.md)。

## 如何运行和验证

安装可选的 pinned PyTorch 依赖后：

```bash
python examples/toy_grpo_train.py
python examples/toy_ppo_train.py
python -m unittest tests.test_ppo tests.test_torch_ppo tests.test_toy_ppo -v
```

`test_ppo` 不依赖 PyTorch：它手算 GAE、clip 和 Critic loss，并验证固定 advantage 时 PPO actor
loss 与 GRPO policy loss 相同。后两组需要 PyTorch，分别验证可微实现/梯度和教学训练的
`pass@1` 确实提升。

2026-08-22 已在 L20 上使用实际 PyTorch（CPU 模式）运行后两类测试：actor/Critic 两路反向
传播与端到端训练均通过，独立 toy run 的 `pass@1` 为 `0.125 → 1.000`。这验证的是教学实现；
真实 4B Critic 校准的运行时状态见 [PPO 可行性记录](../../official_verl/docs/runlogs/2026-08-22-qwen3.5-4b-ppo-gae-feasibility-plan.md)。

## 这项实现的边界与下一步

完成的是**教学级、单 token categorical PPO**：它足以解释 Critic 为什么存在，也能与同环境的
GRPO 做公平的“机制级”对照。尚未完成的是 LLM 的多 token actor-critic batching、真实 value
head、GAE rollout、分布式 Critic 和 4×L20 PPO 资源校准。后者必须先确认显存、吞吐与独立
development/final 划分，不能把这个 toy 直接外推为 4B 训练结论。
