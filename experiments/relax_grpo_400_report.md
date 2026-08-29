# Relax Qwen3-4B GRPO 400-rollout report

更新时间：2026-08-30

这是一条独立的 Relax 数学 GRPO 实验线，不与 Strategy 2 的多轮工具型 Agent RL 结果合并。

## 1. 训练协议

- 硬件：4 x NVIDIA L20 48 GB，Ray 注册 4 张 GPU。
- 模型：Qwen3-4B（`/mnt/storage01/zhangwenchao02/models/Qwen3-4B`）。
- 训练数据：DAPO math 17k。
- 算法：GRPO + DAPO reward score + reference KL（系数 0.01）+ TIS。
- rollout：从 `iter_0000019` 恢复，补跑到总计 400 rollout（最终 `iter_0000399`）；每题 8 个回答，最大 2048 token，temperature 0.8。
- 训练闭环：rollout -> DAPO reward -> group-relative advantage -> actor update -> KL/TIS -> weight sync。
- 结果目录：`/mnt/storage01/zhangwenchao02/exps/relax-qwen3-4b-night-20260829-continue2/`。

训练任务 `raysubmit_jcLHxFkTsLYMmCER` 已成功完成，最终 checkpoint 为 `iter_0000399`。`iter_0000199` 和 `iter_0000399` 均已转换为 HF 权重并通过 SGLang 加载。

## 2. Held-out 评测

数据为 AIME-2024 30 题；base、19-step 和 399-step 使用相同题集、seed `20260829`、temperature 0.8、每题 8 个采样、DAPO 评分器。`eval/aime` 是 `-1/1` reward 的均值，不是准确率；准确率看 `eval/aime/acc/mean`。

### 2048-token（与训练上限一致）

| 模型 | 正确样本 | 单样本准确率 | pass@8 | 截断率 |
| --- | ---: | ---: | ---: | ---: |
| base | 9/240 | 3.75% | 13.33% (4/30) | 83.33% |
| iter 19 | 15/240 | 6.25% | 16.67% (5/30) | 79.17% |
| iter 399 | 19/240 | 7.92% | 20.00% (6/30) | 81.67% |

### 4096-token（长度预算诊断）

| 模型 | 正确样本 | 单样本准确率 | pass@8 | 截断率 |
| --- | ---: | ---: | ---: | ---: |
| base | 16/240 | 6.67% | 36.67% (11/30) | 69.17% |
| iter 19 | 20/240 | 8.33% | 30.00% (9/30) | 60.00% |
| iter 399 | 33/240 | 13.75% | 46.67% (14/30) | 65.42% |

4096-token 逐题配对：12 题提升、17 题持平、1 题退化；每题正确样本比例净增 7.08 个百分点。以 30 题为 bootstrap 单位的探索性 95% 区间为 [3.33, 11.25] 个百分点；这不是多 seed 的显著性结论。

### 4096-token 独立解码 seed 复现

保持 checkpoint、题集和采样协议不变，仅把 rollout seed 从 `20260829` 改为
`20260830`：

| 模型 | 正确样本 | 单样本准确率 | pass@8 | 截断率 |
| --- | ---: | ---: | ---: | ---: |
| base | 21/240 | 8.75% | 33.33% (10/30) | 69.17% |
| iter 399 | 32/240 | 13.33% | 36.67% (11/30) | 62.50% |

第二个 seed 中逐题正确样本比例为 7 题提升、19 题持平、4 题退化，净增 4.58
个百分点。两个解码 seed 合并后，base 为 37/480（7.71%），iter 399 为
65/480（13.54%）；以 30 道 prompt 为聚类单位、先对两个 seed 求平均的探索性
bootstrap 差值为 +5.83 个百分点，95% 区间 [1.25, 11.25]。这增强了“方向可复现”
的证据，但两个 seed 共享题目和同一个训练 checkpoint，不能替代独立训练 seed 或
更大的 held-out 集。

## 3. 结论与边界

1. 工程结论成立：400 rollout 的生成、规则奖励、组相对 advantage、策略更新、KL、TIS、权重同步和 checkpoint 保存均真实完成。
2. 能力结论是正向但有限：399-step 在 2048 和两个 4096 解码 seed 下都超过 base 的单样本准确率；4096 两个 seed 合计从 7.71% 提升到 13.54%。
3. 2048 不是充分的能力评测预算。同一 base 放宽到 4096 后 pass@8 从 13.33% 变为 36.67%，说明长度截断是主要瓶颈。
4. 提升仍集中在少数题目。30 题、两个解码 seed、单一训练 seed 和 checkpoint 仍不足以宣称稳定泛化，也没有证明某个更好的训练配方。
5. 这条线是数学题 GRPO，不是工具型 Agent RL；没有多轮 action -> tool -> observation 环境，也不能用来证明过程奖励有效。

## 4. 后续实验优先级

1. 已补测 `iter_0000199`（4096）：20/240（8.33%）、pass@8 9/30（30.00%）、截断率 60.00%。它没有超过 base 的 pass@8（11/30），也明显低于 `iter_0000399`（33/240、14/30），当前不支持 19-step 早停。
2. 固定评测协议增加独立解码 seed，并扩大 held-out 数量；同时报告准确率、pass@8、截断率和有效答案率。
3. 下一轮训练把“正确终止、长度预算、无效/重复动作”纳入 reward 或采样过滤，验证是否能减少当前约 65% 的 4096 截断。
4. Agent RL 结论继续使用 Strategy 2 的工具型实验单独验证，不与本报告的数学 GRPO 数字混合。

原始评测 JSONL：

- `/mnt/storage01/zhangwenchao02/evals/relax-qwen3-4b-400-20260829/results/{base,iter_0000199,iter_0000399}/eval/0.jsonl`
- `/mnt/storage01/zhangwenchao02/evals/relax-qwen3-4b-400-20260829/results_4096/{base,iter_0000399}/eval/0.jsonl`
- `/mnt/storage01/zhangwenchao02/evals/relax-qwen3-4b-400-20260829/results_4096/iter_0000199/eval/0.jsonl`
- `/mnt/storage01/zhangwenchao02/evals/relax-qwen3-4b-400-20260829/results_4096_seed20260830/{base,iter_0000399}/eval/0.jsonl`

补充评测可用 `experiments/run_relax_grpo_eval.sh` 复现。该脚本启用 Relax 的
`--debug-rollout-only`，只加载 HF 权重进行生成和 DAPO 评分，不创建 actor 或更新参数。
