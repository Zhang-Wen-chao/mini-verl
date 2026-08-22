# 训练 reward 的安全有理数归一化审计（2026-08-22）

## 结论

Qwen3.5-4B 的 679-step 正式 GRPO run 使用的上游 `math_reward` 只比较规范化后的字符串；它会将数学等价的小数和分数误判为不同答案。现已实现一个**可选、保守**补丁：保留 legacy scorer 的所有命中，只在最终 `\boxed{...}` 答案和 ground truth 都能无歧义解析为有限有理数时，额外判定相等。

完整历史 rollout 审计将正样本由 **1,685/8,148** 提高到 **1,697/8,148**：新增 **12** 条正确奖励（**0.147%**），影响 **7/2,037** 个 prompt group。其中仅 **3** 个 group 从全同奖励变为 mixed reward group，因而首次获得非零 GRPO 相对 advantage。

这证明修复应纳入后续训练和评测的正确性契约；但覆盖率过低，不能把一整夜的 170-step A/B 质量差异诚实地归因于它。因此它不是本夜的主算法变量。

## 范围与安全边界

- 支持：整数、有限小数、`a/b`、`\frac{a}{b}` / `\dfrac{a}{b}`；使用精确 `Fraction`，没有浮点近似。
- 不支持：方程、集合、元组、根式、变量表达式、单位或一般符号等价；这些全部回退到 upstream legacy scorer。
- 前导零整数（例如 `09`）保持 legacy 行为，因为它可能表示复合答案而不是数字 9。
- 通过 `MINI_VERL_MATH_REWARD_MODE=normalized` 显式启用 import hook；默认运行仍使用锁定的 upstream scorer，未修改 pinned VeRL 源码。

## 可复查证据

- 输入：正式 artifact 的 679 个 JSONL rollout、共 8,148 条样本。
- 远端审计报告：`artifacts/reward-audits/2026-08-22-normalized-rational-audit-679step.json`；SHA-256 为 `3ce82ea1064ddffdcb8e5bfc9db6047bab3e68d7cb4da9d321061e0ee86b9375`。
- group effect：legacy mixed / degenerate = 485 / 1,552；normalized = 486 / 1,551；mixed-status changed = 3。
- 典型修复：`\boxed{5.25}` 对 `\frac{21}{4}`、`\boxed{1/10}` 对 `0.1`、`\boxed{0.80}` 对 `0.8`。

## 实现与验证

- [compat/normalized_math_reward.py](../../official_verl/compat/normalized_math_reward.py)：严格的有理数 fallback。
- [compat/sitecustomize.py](../../official_verl/compat/sitecustomize.py)：只在 opt-in 环境变量下包裹 `verl.utils.reward_score.math_reward`。
- [audit_normalized_math_reward.py](../../official_verl/audit_normalized_math_reward.py)：对保存的历史 `score`（实际训练 reward）与新 scorer 做审计，不依赖重跑 VeRL。
- `python3 -m unittest tests.test_normalized_math_reward -v`：4 项通过，覆盖小数/分数、嵌套 fraction、错误答案、无 box、最后一个 box 和前导零边界。

## 后续决策

完整训练应启用此修复并记录审计；若要声称 reward 修复带来质量提升，需要在它启用后独立运行足够长、且采用预先冻结开发/最终测试集的实验。当前历史数据不支持把它作为高功效质量对照。
