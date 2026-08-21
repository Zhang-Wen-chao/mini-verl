# Qwen3.5-4B 正式 run 的 one-step 时序审计（2026-08-22）

## 结论

正式 679-step GRPO run 使用的是**受控 one-step 异步**：训练 batch `k` 时，rollout
侧可以生成下一批；但权重同步、reference 与 actor update 的计时并非互斥。因此优化与
容量判断不能把所有 `timing_s/*` 数字简单相加。

日志中有 87 条带完整细粒度 telemetry 的 step 记录（步骤 1--87）；后续 592 步保留了训练
结果与 artifact，但没有相同的一组细粒度 timing 字段。本报告只对这 87 步做系统性能结论。

| 指标（87 步，秒） | mean | p50 | p90 |
|---|---:|---:|---:|
| 端到端 `timing_s/step` | 79.99 | 76.38 | 93.02 |
| `generate_async` | 13.56 | 13.00 | 16.99 |
| `sync_rollout_weights` | 29.93 | 29.66 | 33.28 |
| `ref` | 13.68 | 12.99 | 17.18 |
| `update_actor` | 36.29 | 33.49 | 46.09 |
| 上述四项的**包含式**计时和 | 93.47 | 89.64 | 106.94 |

四项 mean 的和比端到端 step 多 **13.48s**。这不是未解释的开销，而是重叠工作的可见
证据：它们是各自操作的 inclusive time，不能被解读为串行分解。

## 能指导的优化方向

1. **优先 weight sync。** 同步约 29.9s，是比 rollout generation 更稳定、更大的单项。
   下一轮 profile 应先测传输路径、bucket 设置、同步频率/压缩的可行性；不得破坏 policy
   version 契约或放宽 one-step lag。
2. **其次 actor/ref。** actor update 均值 36.3s、p90 46.1s，reference 约 13.7s；其成本
   随 token 长度波动，值得结合 sequence packing、长度分布和 FSDP profile 分析。
3. **不要先优化 rule reward。** 保存的 agent-loop 单样本 score 约 10ms；训练主链路中的
   reward 计时近乎零，对约 80s step 的端到端吞吐没有实质贡献。
4. **rollout 并非当前首要瓶颈。** `generate_async` 约 13.6s，且已与后续工作重叠。单独让
   vLLM 更快未必线性缩短 step，必须重新测 wall-clock。

## 运行特征与边界

- mean response length 174.36 token；response cap ratio mean 12.36%，p90 33.33%。长度仍是
  训练成本与质量的共同变量，不能只看 samples/s。
- actor MFU mean 0.90%，说明这个 3 FSDP + 1 rollout 的小 batch 系统主要受编排、同步、
  内存/通信和变长序列限制，不能用大模型单机 MFU 期望直接评价。
- 此报告不推出 679-step 全程的精确平均 step time，也不声称任何单项是完全可独立消除的
  latency；它只提供有 telemetry 覆盖的可复查系统归因。

## 可复查证据

- 输入：正式 artifact 的 `logs/train.log`。
- 远端报告：`artifacts/timing-audits/2026-08-22-formal-679step-one-step-timing.json`；SHA-256
  `f2d28b33bd1eeda2f900b233a3e601800f369ab21afbe6cda0e3dfba6898d9af`。
- [analyze_one_step_timing.py](../../official_verl/analyze_one_step_timing.py) 移除 ANSI 前缀后
  解析完整 telemetry；其单测验证 ANSI 前缀、字段提取和 overlap 差值。
