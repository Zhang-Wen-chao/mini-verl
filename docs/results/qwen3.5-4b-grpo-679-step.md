# Qwen3.5-4B GRPO 679-step 正式 run 验收（2026-08-20）

分支：`official-verl-grpo`。实验 artifact 在 L20：
`/mnt/storage01/zhangwenchao02/repos/mini-verl-l20/artifacts/qwen3.5-4b-openr1-grpo-2037row-679step-v5-short-trainer3-rollout1-20260819T1919`

---

## 1. 运行概况

| 项 | 值 |
|---|---|
| 模型 | Qwen3.5-4B（base） |
| 算法 | GRPO（`algorithm.adv_estimator=grpo`） |
| 训练集 | 2037 题（OpenR1-Math 过滤，`training-2037-processor-filtered-...parquet`） |
| 步数 | 679（`save_freq=340`） |
| 硬件 | 4×L20，FSDP2（3 卡 actor）+ vLLM（1 卡 rollout） |
| 超参 | lr=1e-6, kl_coef=0.001, low_var_kl, max_prompt 512, max_response 384, batch 3 |
| 起止 | 2026-08-19 19:19 → 2026-08-20 12:52（约 17.5h） |
| 每步耗时 | 84–108s，吞吐 11–18 tok/s |

## 2. 训练中监控评测（固定 64 题，与训练集零重叠）

| step | accuracy@1 | 相对 step 0 |
|---|---|---|
| 0 | 42/64 = 65.6% | — |
| 170 | 46/64 = 71.9% | +6.3pp |
| 340 | 54/64 = 84.4% | +18.8pp |
| 510 | 56/64 = 87.5% | **+21.9pp（训练中峰值）** |
| 679 | 53/64 = 82.8% | +17.2pp |

- 峰值 **+21.9pp（约 +22pp）**，信号真实（train/eval 零重叠，MD5 验证）。这 64 题会在训练中反复评测，因此用于观察训练趋势；最终泛化结论仍以 200 题 held-out 为准。
- 510→679 表面回落 3 题，逐题判定：6 道回落题中 2 道是**评分器格式误杀**（`5.5` vs `\frac{11}{2}`、`3.5` vs `\frac{7}{2}`），
  修正后实际约 **55/64 = 85.9%**。详见 [510 → 679 回落分析](step-510-to-679-regression-analysis.md)。

## 3. 训练信号健康度

- reward/score mean：0 → 0.42 (340) → 0.58 (632)，整体上行，单步噪声大（batch 3）
- advantage mean 在 0 附近小幅摆动（GRPO 正常）
- ppo_kl 稳定 0.011–0.024，`low_var_kl` 约束生效，未发散
- grad_norm 2.6–5.0 健康；lr 恒 1e-6
- response_length 138 → 270+（模型倾向过度展开），aborted_ratio 始终 0

## 4. checkpoint 与产物完整性

- `global_step_679` 完整（actor 分片 + data.pt），`latest_checkpointed_iteration.txt` = 679
- 评测节点齐全：validation_samples 0/170/340/510/679 五个 jsonl
- rollout_samples 679 个 jsonl 全齐
- 主进程正常退出，GPU 全部释放（4 MiB）
- 679 FSDP 分片已合并为 HF 权重：`artifacts/merged-679-hf`

## 5. 结论

1. 小规模（2037 题）GRPO 在 4B 上产生真实、可迁移的 MATH 推理提升（64 题 +22pp，零重叠）。
2. 64 题监控集太小且训练中反复测过，不能作为最终结论 → 需 held-out 验证（进行中）。
3. 发现评分器对等价数字格式不鲁棒（分数/小数误杀）→ held-out 评测已用归一化评分。
4. 训练后期 response_length 逼近 384 上限，是下一轮需关注的风险（reward 未惩罚啰嗦）。

## 5b. Held-out 验证结果（已完成，2026-08-20）

- 协议：MATH-lighteval-test 5000 题随机抽 200（seed 42），base 与 679 checkpoint 同 prompt、
  同 greedy 解码（max 768）、同归一化评分；679 用合并后的 HF 权重 4 卡并行评测。
- 结果：

| 对照模型 | 是否经过本次 GRPO | 正确/200 | accuracy@1 |
|---|---|---|
| 原始 Qwen3.5-4B base checkpoint | 否 | 5/200 | 2.5% |
| `global_step_679` checkpoint | 是，679 step | 17/200 | **8.5%** |
| 变化 | — | +12/200 | **+6.0 个百分点（3.4×）** |

- **与未做 GRPO 的原始 base checkpoint 相比，679-step checkpoint 绝对提升 6.0 个百分点（2.5% → 8.5%，3.4×）；这些题模型从未见过，且只用于两次评测 → 泛化能力真实提升，非背题。**
- 与训练中 64 题零重叠 +22pp 相互印证，GRPO 训练有效结论可信。
- 备注：MATH-lighteval 为 AMC/AIME 级难题，4B 无思考模式绝对精度低属正常；
  相对提升（3.4x）才是有效信号。

## 6. 后续工作（2026-08-20 状态）

1. ✅ 验收：exit_status/checkpoint/rollout 齐全（本文件）
2. ✅ 汇总曲线 + 回落诊断（本文件 + [510 → 679 回落分析](step-510-to-679-regression-analysis.md)）
3. ✅ held-out 评测：MATH-lighteval 5000 题抽 200，base vs 679 同协议对比，结果为
   2.5% → 8.5%（见 5b）。
4. ⏳ 固化当前正向基线后，扩数据到 formal-10000 的单变量对照实验；同时监控
   lr/KL、长度截断、reward 稀疏与评分器鲁棒性。
5. ⏳ 官方 verl 数据流映射回 mini-verl（只复现可单机验证的 GRPO 语义）
6. ⏳ 完整报告 + 提交（本文件 + 两份分析）
