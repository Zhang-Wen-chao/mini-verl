# Strategy 2 Agent RL 最终评测报告

更新时间：2026-08-27

这份报告记录 `agent-rl` 分支上的 Strategy 2 follow-up。它与
[`experiments/overnight_report.md`](../overnight_report.md) 的早期 Qwen3-0.6B
本地 smoke 是两组不同实验，不能合并或取平均。

## 1. 评测协议

- 模型：Qwen3-4B-Instruct-2507；v3 从 base 重新训练，不续训 v2。
- 数据：AIME-2024 held-out 30 题；训练用 DAPO math 17,398 条，题面审计与评测集无直接重合。
- 解码：`temperature=0`、`max_new_tokens=1024`、最多 16 轮工具调用。
- 评测修复：最终答案提取排除 tool call、code 和 tool observation；recovery prompt 中的 `Answer: \\boxed{answer}` 占位符不参与评分；支持真实裸 `\\boxed{...}` 输出。
- 结果根目录（远端，不把 records/checkpoint 放进 Git）：`/mnt/storage01/zhangwenchao02/strategy2-overnight-followup-20260826T010200`。
- 远端汇总：该目录下的 `comparison.json`；每个 arm 均有 30 条 `records.jsonl` 和 `summary.json`。

## 2. 8K 协议修复后的四策略正式对照

| Arm | 正确数 | 答案终止 | 平均工具调用 | 平均无效动作 | 工具成功率 | 平均输出 token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 8/30 (26.7%) | 21/30 | 3.13 | 2.67 | 64.9% | 3,634 |
| outcome reward | 6/30 (20.0%) | 20/30 | 3.33 | 2.00 | 78.0% | 3,707 |
| process reward | 7/30 (23.3%) | 21/30 | 3.53 | 1.97 | 60.4% | 3,210 |
| quality-process v2 | 6/30 (20.0%) | 15/30 | 5.33 | 3.03 | 58.8% | 4,306 |

与 base 的逐题配对结果：

| Arm | Arm 独对 | Base 独对 | 共同正确 | 共同错误 | 正确数差 | 精确 McNemar 双侧 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| outcome reward | 2 | 4 | 4 | 20 | -2 | 0.6875 |
| process reward | 3 | 4 | 4 | 19 | -1 | 1.0 |
| quality-process v2 | 3 | 5 | 3 | 19 | -2 | 0.7266 |

结论：单一 seed、30 题下，没有 reward arm 显著优于 base。v2 的主要退化不是只看正确率：它平均工具调用从 3.13 增至 5.33，平均无效动作从 2.67 增至 3.03，答案终止从 21/30 降至 15/30。

## 3. 16K context 容量消融

这是对同一批已有 checkpoint 的上下文容量对照，不是新的训练实验。

| Arm | 8K 正确数 | 16K 正确数 | 16K 平均工具调用 | 16K 答案终止 |
| --- | ---: | ---: | ---: | ---: |
| base | 8/30 | 7/30 | 4.17 | 19/30 |
| outcome reward | 6/30 | 6/30 | 3.97 | 20/30 |
| process reward | 7/30 | 7/30 | 4.40 | 20/30 |
| quality-process v2 | 6/30 | 7/30 | 5.40 | 19/30 |

同 checkpoint 的逐题配对检验均为 `p=1.0`。结论是：16K 没有带来稳定正确率收益；它改变了可容纳的长轨迹数量，但不能替代更好的 reward 或终止协议。

## 4. 修复协议后的 v3 对照

v3 从 base 重新短训，唯一实质变化是修复后的答案/恢复协议与 quality-process reward。8K paired held-out 结果：

| Arm | 正确数 | 平均工具调用 | 平均无效动作 | 答案终止 | 平均输出 token |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 7/30 | 3.23 | 2.83 | 19/30 | 3,904 |
| quality-process v3 | 8/30 | 3.90 | 2.97 | 17/30 | 3,720 |

逐题配对为：共同正确 5 题、base 独对 2 题、v3 独对 3 题、共同错误 20 题；正确数净增 1 题，精确 McNemar 双侧 `p=1.0`。

结论：v3 没有复现 v2 的明显正确率退化，出现一个弱正向信号，但工具调用更多、答案终止更低，不能说 quality-process reward 已经被证明有效。它只值得作为独立训练 seed 的复现候选。

## 5. 最终结论与边界

1. 评测协议 bug 已修复，三组评测共 10 个 arm（8K 四策略、16K 四策略、8K v3 paired）全部完成，每个 arm 都是 30/30 记录并生成 `summary.json`。
2. 这轮实验没有统计显著 winner。最稳妥的结论是：当前 outcome/process/quality-process 配方都没有在这个小规模 held-out 上证明优于 base。
3. quality-process v2 暴露了典型 reward hacking/终止问题：模型更愿意调用工具，但没有同步提高有效求解和及时结束。
4. v3 说明修复答案协议是必要条件，但协议修复本身不等于 reward 有效；下一步必须预注册独立训练 seed、扩大 held-out 题数，并同时验收正确率、答案终止率、无效动作和工具调用成本。
5. 这份报告只记录实验结果，不提交远端 checkpoint、模型副本或逐条 records；复现实验使用 `experiments/strategy2/run_heldout_eval.sh` 和 `run_overnight_followup.sh`。
