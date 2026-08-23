# 全自研栈：mini-verl + mini-megatron + mini-vllm 的 GRPO（2026-08-23）

## 目标

验证"从 RL 算法到推理引擎到并行训练框架全部自研"的完整闭环是否可行：
用自研组件替代官方 verl / vLLM / Megatron 栈，跑通并证明 GRPO 真的有效。

## 架构

```
mini-verl（RL 编排，自研）
  ├─ rollout ← mini-vllm（自研 paged-attention 推理引擎）
  │     └─ TransformersAdapter 驱动 mini-megatron GPT（无 config 时从结构推断）
  └─ train ← mini-megatron GPT（自研并行 transformer）
        └─ MiniMegatronTrainerAdapter（forward(input_ids, attention_mask) -> .logits）
```

- 推理：mini-vllm 的 KV cache / scheduler / 采样，加上 TransformersAdapter
  用 forward-hook 抓 K/V 存入 BlockTable。
- 训练：mini-megatron 的 GPT（单卡模式 TP=1），经适配器满足 mini-verl
  trainer 接口。
- 采样：mini-vllm 新增 temperature/top-p（原引擎只有 greedy）。

## 关键修复（跑通 → 有效的 3 个真实 bug）

1. **preempt 死循环**（engine `_make_room_for_next_tokens`）：KV 池无法腾出块时
   preempt→重新 admit→prefill 无限循环。修复：限定 preempt 次数，给不出块就
   放弃本轮。
2. **max_prefill_tokens 太小**：MATH 长 prompt（173 token）超过 128 后 scheduler
   永远拒绝 admit，`while has_requests()` 死循环。修复：按真实 prompt 长度配置。
3. **old_logprobs 占位 0**（关键）：rollout 填 `(0.0,) * len`，导致 GRPO 的
   `ratio = exp(new_logprobs - 0) = exp(new_logprobs)` 几乎总低于 0.8 clip 下限
   → clip_fraction=1.0，策略永不更新（"跑通但学不动"）。修复：rollout 后
   用模型对完整序列（prompt+response）前向，重算真实 per-token logprobs。

## 有效性验证（公平对比）

设计：同一初始权重（seed 42）、同一 prompt、同一采样参数；唯一变量是
GRPO 更新。基线 = 初始模型直接 rollout 测准确率；实验组 = 同权重模型
GRPO 训练 80 步后测同一任务。

任务：响应中必须包含至少一个偶数 token（2/4/6/8）——31K toy GPT 可学
（第一版"以 7 结尾"太稀疏，reward≈0 无正样本，验证失败，遂调整任务难度）。

| 指标 | 基线（未训练） | GRPO 训练后 | 提升 |
|---|---|---|---|
| 准确率 | 0.713 | **0.975** | **+0.262** |
| 训练 reward | 0.75 | 1.00 | 单调上升至满 |
| clip fraction | - | ~0.000 | 正常（修复后） |

结论：**自研 GRPO 在自研栈上真实优化了策略**（+26pp，非噪声）。
脚本：`examples/grpo_effectiveness.py`（mini-megatron 仓库）。

## 相关提交

- mini-vllm: 989c826（采样）、29a220f（TransformersAdapter）、7cc8b2c（preempt 修复）
- mini-verl: 3996eda（mini-vllm rollout 后端）、1884a59（old_logprobs 修复）
- mini-megatron: 84e444a（全自研栈 GRPO）、822932f（有效性验证）

## 局限与后续

- 当前 decode 重算全序列（未利用 paged K/V），慢；后续可让 decode 读 BlockTable。
- 验证用 toy 模型（31K），真实模型（Qwen3-0.6B）链路已通（见 3996eda）但
  有效性对比待做。
- PPO 可用同一链路验证（mini-verl 有 ppo.py，需带 critic）。
