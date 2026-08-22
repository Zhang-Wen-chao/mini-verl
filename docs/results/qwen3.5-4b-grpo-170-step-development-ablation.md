# Qwen3.5-4B GRPO advantage 标准化：170-step 开发对照（2026-08-22）

## 结论

完成了一组从同一 Qwen3.5-4B base 出发的真实 4×L20 受控筛选：先通过 20-step
no-std calibration，再运行 no-std GRPO 与 standard GRPO 各一个 170-step development
run。两段长实验均完成 170 rollout、`global_step_170`、训练 exit status `0` 与 watcher
exit status `0`；未见 OOM、NCCL、NaN 或 Inf 训练失败。

这证明两种 GRPO advantage scaling 配置在当前官方 `verl` 3 trainer + 1 vLLM rollout
链路中都健康可运行。它**不**证明 standard GRPO 已经优于 no-std：这只是单 seed，且
固定 64 题 development monitor 的起点不同；MATH held-out 200 在本次实验中完全没有读取。

## 唯一算法变量与固定契约

| 项目 | 固定值 |
|---|---|
| 模型 / 起点 | `Qwen3.5-4B` base；两个 170-step run 均从 base 启动，不续训 calibration |
| 训练数据 | 去重后的 OpenR1-Math 2,037 rows |
| 拓扑 | 4×L20；3 FSDP2 trainer GPU + 1 TP=1 vLLM rollout GPU |
| rollout | `n=4`，temperature `0.8`，top-p `0.95` |
| 优化 | LR `1e-6`；reference KL `0.001`；legacy math rule reward |
| 序列上限 | prompt 512 / response 384 / model context 896 |
| 固定监控 | 64 rows；仅作 development monitor |
| 唯一预定变量 | `algorithm.norm_adv_by_std_in_grpo=false`（no-std）或上游默认 `true`（standard） |

两段 run 的监控起点实际为 42/64 与 41/64，说明即使契约相同，完整生成/服务链也没有
提供可把一次 monitor 差异解释为纯算法效应的严格复现实验。因此比较时只将其视为筛选
线索，而不是估计无偏算法效应。

## 完成与 telemetry

| Run | 64 题 monitor：step 0 → 170 | rollout | mixed reward group 且 grad non-zero | 有限 KL metric steps | 平均 step time |
|---|---:|---:|---:|---:|---:|
| no-std GRPO | 42/64（65.6%）→ 47/64（73.4%），+7.8pp | 170 | 121/121 | 170 | 22.20 s |
| standard GRPO | 41/64（64.1%）→ 48/64（75.0%），+10.9pp | 170 | 126/126 | 170 | 22.29 s |

两段速度几乎相同。最后一个训练 step 的数据为：no-std `KL=0.0112`、
`grad_norm=1.96`、response clip ratio `0.583`；standard `KL=0.0169`、
`grad_norm=0.0105`、response clip ratio `0.417`。单个 final batch 受 rollout 随机性影响，
不应据此比较稳定性；所有 170 个记录的 KL 均为有限数。

## calibration gate

在启动长 run 前，20-step no-std calibration 已成功完成：20 rollout、
`global_step_20`、两个 exit status `0`。gate 检查了完整 checkpoint、rollout 数、
有限 KL、mixed reward groups 的非零梯度、OOM/NaN/Inf 标记，以及最近 response cap
是否灾难性退化。该 calibration 本身不用于质量结论。

## 产物与可复查性

远端 L20 产物（不提交到 Git；每个长 run 约 54 GiB）为：

```text
/mnt/storage01/zhangwenchao02/repos/mini-verl-l20/artifacts/
  qwen3.5-4b-openr1-grpo-nostd-20step-trainer3-rollout1-20260822T0306
  qwen3.5-4b-openr1-grpo-nostd-170step-trainer3-rollout1-20260822T0306
  qwen3.5-4b-openr1-grpo-standard-170step-trainer3-rollout1-20260822T0306
  qwen3.5-4b-openr1-grpo-paired-development-summary-20260822T0306.md
```

本仓库中的控制与验收工具：

- [no-std calibration gate](../../official_verl/await_no_std_calibration.sh)
- [no-std 170-step gate](../../official_verl/await_no_std_development_run.sh)
- [standard 170-step gate](../../official_verl/await_standard_grpo_control_run.sh)
- [配对汇总器](../../official_verl/summarize_grpo_development_pair.py)

汇总器只读取两段 run 的日志、checkpoint、rollout count 与 64 题 monitor；代码不加载或
评分 MATH held-out 200。

## 下一步：避免污染评测

1. 不用本次 64 题 monitor 宣布算法 winner，也不以 MATH held-out 200 选择算法。
2. 若继续，需要先固定新的 development / final 划分、seeds、预算与停止规则。
3. 只有多 seed 开发筛选给出一致信号后，才考虑对最终候选做一次受保护的 held-out 评测。
4. 数据规模、数学 reward 归一化、response-length 目标应各自作为独立变量，而不是与
   advantage 标准化混在同一次长训练里。
