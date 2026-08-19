# Qwen3.5-4B 官方 verl GRPO 计划

状态：**官方 4-GPU 训练链路、单步有效 GRPO 更新与 3+1 两步稳定性均已验收；
尚未进入 2,048/10k 短训。**

> 2026-08-19 最新门槛：原始 256-token、512-token 及“只输出答案”配置均不能
> 给 GRPO 产生可用的组内差异。冻结的真实 OpenR1 rollout 证明“最多三句短解答
> + `\boxed{}`、thinking off、384 response token”在固定题组上产生了
> `[0,0,0,1]`。这只是可进入**一条**官方 GRPO 校准的必要条件，不是训练质量
> 结论。完整证据见 `RUNLOG_2026-08-19_qwen3_5_4b_reward_diagnosis.md`。

这份计划承接已完成的 Qwen3-0.6B 官方 verl 系统 smoke。那次运行已经验证
了官方 `verl`、FSDP2、TP=2 vLLM、GSM8K 规则奖励、checkpoint 和 clean exit
可以在 4 张 L20 上连通；它没有验证学习质量：held-out 结果为 0/64，且最后
一步 84.375% 的回答撞到 256-token 上限。因此 4B 的首要目标不是立刻长训，
而是先让奖励契约和可观测性成立。

## 已核实的候选

2026-08-19 查询 Hugging Face 官方 `Qwen/` 命名空间的结果如下。

| 官方模型 | 是否公开 | 适合当前 4B 实验 | 结论 |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-4B` | 是，Apache-2.0，Transformers | 是 | **首选** |
| `Qwen/Qwen3.5-4B-Base` | 是，Apache-2.0，Transformers | 仅作 base/instruct 对照 | 不作为第一条质量路径 |
| `Qwen/Qwen3.5-9B` | 是，Apache-2.0，Transformers | 可作为 4B 后扩展 | 暂不选 |
| `Qwen/Qwen3.6-27B` | 是 | 参数量不是 4B | 暂不选 |
| `Qwen/Qwen3.6-35B-A3B` | 是，MoE | 参数/并行变量更多 | 暂不选 |
| `Qwen/Qwen3.8-27B` | 是 | 参数量不是 4B | 暂不选 |
| `Qwen/Qwen3.8-2.4T-A95B` | 是，MoE | 远超当前单机实验范围 | 暂不选 |

因此，“Qwen3.5 之后有没有官方 4B”这个问题的答案是：官方 Qwen3.5 有
4B；Qwen3.6 和 Qwen3.8 的官方发布线没有 4B。搜索到的 Qwen3.8-4B 是
第三方发布，不能替代官方受控实验输入。

`Qwen/Qwen3.5-4B` 的 Hugging Face 元数据标记为 conversational、
image-text-to-text，并指向 `Qwen/Qwen3.5-4B-Base`。本实验只使用它的文本
聊天路径；在下载前必须在锁定的官方 verl runtime 中验证其 Transformers model
type、chat template、vLLM 0.24.0 加载和 rollout log-prob 路径。不能仅因模型
页面标有 Transformers 就假定当前 pinned verl 组合兼容。

## 固定与变化

保持不变：

- 上游 verl revision：`c4b389adadc58ce51cb2b63e70df497ca166d77f`。
- 锁定 runtime：同一 `uv.lock`、FSDP2 actor/reference、vLLM rollout、Ray。
- 拓扑：资源校准默认用 GPU 0--1 FSDP2 训练、GPU 2--3 TP=2 vLLM；通过
  3 trainer + 1 TP=1 rollout 的受控两步实验后，后续 4B 短训优先使用后者。
- 算法：GRPO；奖励首先仍使用可验证的数学最终答案规则奖励。
- 运行前安全检查：确认四张 L20 的显存、利用率和 compute process，绝不抢占
  他人作业；显式设置 `NCCL_SHM_DISABLE=1` 与
  `CUDA_DEVICE_MAX_CONNECTIONS=1`。

只在逐个通过关卡后改变：模型从 0.6B 到 4B、数据规模、response 上限、训练
步数和实际优化器批量。训练后端、推理后端、模型、数据、奖励格式不能在同一轮
一起更换。

## 阶段 A：兼容性与资源 smoke

用户已授权并完成下载。不可变输入位于 L20 的个人持久盘（不是容器 `/tmp`）：

| 输入 | 持久路径 | 已验证的身份 |
| --- | --- | --- |
| 模型 | `/mnt/storage01/zhangwenchao02/repos/mini-verl-l20/.official-verl/models/Qwen3.5-4B` | `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；两个权重 shard 的 SHA-256 分别为 `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61` 与 `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188` |
| 训练源 | `/mnt/storage01/zhangwenchao02/repos/mini-verl-l20/.official-verl/data/OpenR1-Math-220k-default-rev-e4e141ec` | `open-r1/OpenR1-Math-220k`, `default`, revision `e4e141ec9dea9f8326f4d347be56105859b2bd68`；10 个 parquet、93,733 行，目录内有 `SHA256SUMS` |
| MATH 评测 | `/mnt/storage01/zhangwenchao02/repos/mini-verl-l20/.official-verl/data/qwen3_5_4b/MATH-lighteval-test` | `DigitalLearningGmbH/MATH-lighteval`, `default/test`, revision `0530c78699ea5e8eb5530600900e1f328b48acad`；5,000 行 |
| GSM8K 评测 | `/mnt/storage01/zhangwenchao02/repos/mini-verl-l20/.official-verl/data/qwen3_5_4b/gsm8k-test` | `openai/gsm8k`, revision `740312add88f781978c0658806c59bc2815b9866`；1,319 行 |

远端输入总清单为
`/mnt/storage01/zhangwenchao02/repos/mini-verl-l20/.official-verl/data/qwen3_5_4b/INPUTS_MANIFEST_2026-08-19.txt`。
锁定 runtime 已能在纯本地文件模式加载模型配置和 tokenizer：`model_type=qwen3_5`、
架构为 `Qwen3_5ForConditionalGeneration`，并且存在 chat template。它是多模态
conditional-generation 架构，因此仍须在阶段 A 用当前 pinned vLLM 做实际文本
rollout、log-prob 与权重同步检查；不能把 tokenizer 可加载误报为 verl 完全兼容。

1. 用 pinned runtime 对 `Qwen/Qwen3.5-4B` 做单个 prompt 的 tokenizer/chat-template
   和 vLLM 加载检查；确认文本生成、log-prob 和权重同步路径均可用。
2. 依照 0.6B 的官方脚本另建 4B 配置，首先用 32--128 条固定样本、2--4 个
   optimizer steps。重新估算并实测 FSDP actor/reference、optimizer state、
   activations 及 TP=2 vLLM KV cache 的显存，不能照抄 `gpu_memory_utilization=0.60`
   或 micro batch。
3. 验收：locked preflight、4-GPU launch、至少一个 checkpoint、restore、clean
   exit status 均成功；记录每卡 peak memory、tokens/s、step time。

阶段 A 的目标只是“4B 能稳定跑”，不报告模型变好。OOM、模型结构不兼容或
rollout/log-prob 不通时，先修兼容配置，不进入下一阶段。

## 阶段 B：奖励与数据契约校准

### 已选数据契约

首轮 4B 质量实验使用一套训练源和两套互不参与训练的评测：

| 角色 | 数据 | 为什么选它 | 规模与已锁定的 Hub revision |
| --- | --- | --- | --- |
| 训练 | `open-r1/OpenR1-Math-220k`，`default` config | Apache-2.0；直接提供 `problem`、`answer`、题型和来源字段，适合做可审计的数学规则奖励与分层抽样 | 93,733 题，下载约 2.15 GB；`e4e141ec9dea9f8326f4d347be56105859b2bd68` |
| 主评测 | `DigitalLearningGmbH/MATH-lighteval`，`default/test` | MIT；5,000 道独立的竞赛数学题，`solution` 的最终答案为 `\boxed{...}`，比 GSM8K 更能观察难题泛化 | 5,000 题；`0530c78699ea5e8eb5530600900e1f328b48acad` |
| 连续性评测 | 已有的 GSM8K test | 与 0.6B smoke 保持可比，诊断简单整数格式是否仍然失败 | 不进入训练 |

这不是把多个训练集混合：**训练只来自 OpenR1 default**。MATH 和 GSM8K 只作为
冻结评测，绝不在 rollout/更新中使用。`Qwen/Qwen3.5-4B` 已选定；模型和数据的
最终下载 revision 在实际下载完成后还必须用文件 SHA-256 固化。

OpenR1 的 `answer` 已是独立字段，这是它优先于 NuminaMath-TIR 的关键：后者虽然
较小（72,441 题、约 148 MB、Apache-2.0），但只有 `problem`/`solution`，需要另写
并验证最终答案抽取器。GSM8K 则保留为 smoke/continuity benchmark，而非 4B 主训练
语料。

在转换为 verl parquet 前必须执行一次 leakage audit，保存审计报告：

1. 依据 OpenR1 的 `source` 字段排除所有显式标为 MATH、GSM8K 或所选评测源的
   训练行。
2. 对规范化后的题面与 MATH test、GSM8K test 做 exact de-dup；命中项从训练中剔除。
3. 记录过滤前后数量、命中原因和每个评测集的 SHA-256；若无法证明独立性，MATH
   分数只能标为参考，不能声称泛化。

先冻结一个 2,048 题的确定性、按题型分层的 calibration 子集；阶段 C 再使用固定
的 10,000 题训练子集。两者均从上述审计后的训练池产生，使用记录在 manifest 中的
seed；不直接以 93k 全量开始。这样先验证 reward 和显存，再把运行成本增加到足以
观察趋势的规模。

数学 reward 不再沿用 GSM8K 的“`#### <integer>`”解析器：训练和 MATH 评测统一要求
回答在 `\boxed{...}` 中给出最终答案。当前首轮使用 pinned verl 自带的
`math_reward`：它取最后一个 boxed answer，并做受控的 LaTex 字符串归一化
（例如分数写法），**不是**完整的符号等价证明器。若后续需要更强的 Math-Verify
符号比较，必须先获得新增依赖的授权，不能静默 `pip install`。

当前 pinned verl 的 rule-reward router 不认识 `open-r1/OpenR1-Math-220k` 这个
键；但它认识 `DigitalLearningGmbH/MATH-lighteval`，并将其路由到上游的
`math_reward`（最后一个 `\boxed{...}` + LaTex 归一化）。因此转换后的
`data_source` 固定为后者，仅作为**评分器选择键**；原始训练来源仍以
`extra_info.raw_data_source=open-r1/OpenR1-Math-220k` 与 audit manifest 保存。
不能把这两个概念混为一谈，否则训练要么无法路由 reward，要么丢失数据谱系。

必须补上 0.6B run 缺失的可观测性：为固定 prompt 子集保存原始 prompt、每个组内
的 completions、解析到的答案、reward、parse/error reason、response token length
及 policy version；不得保存密钥或无关用户数据。

在**不更新权重**的 rollout calibration 中调 prompt/chat template、终止格式、
temperature、group size 和 response 上限。优先排查 `#### <integer>` 契约是否与
Qwen3.5 的聊天回答吻合，以及 max response length 是否截断最终答案。

进入短训前必须同时满足：

- 原始样本能人工抽查，答案解析与规则奖励可解释。
- reward 在多个 batch 中不是全 0、不是全 1；同 prompt 组内至少偶尔有差异，
  这样 GRPO 才会产生有意义的 relative advantage。
- response-cap ratio 不再像 0.6B run 那样长期接近 1。
- base held-out accuracy、reward histogram、答案解析失败率和长度分位数已保存。

若 base 模型在严格 `####` 契约上全零，不能靠增加训练步数解决；先修 prompt/
verifier/上限。首选聊天模型正是为了降低“base 模型不遵守作答格式”这一变量，
但这仍须由样本证明。

## 阶段 C：正式短训

阶段 B 固定后，执行一次可复现实验：固定模型 revision、数据子集、seed、配置、
基线评测和 held-out 集；仅增加训练步数。保存 checkpoint、launcher exit status、
原始样本审计、训练曲线和最终评测。

报告至少包含：

- base 与 final held-out accuracy/reward；
- reward distribution、parse failure rate、response-length 分布和 cap ratio；
- actor loss、KL、gradient norm、step time、tokens/s 和各 GPU peak memory；
- 与基线的差异及不确定性，而不是只报告训练 reward。

只有出现非退化奖励、干净完成和相对于同一基线可解释的 held-out 变化，才可称为
“4B 质量实验有结论”。没有提升也是有效结论，只要证据完整。

## 后续对照，不与首轮混做

1. 在同一模型、数据、seed 与资源预算下比较 vLLM 和 SGLang rollout。
2. 在同一实验契约下比较 FSDP2 与 Megatron；这时才有资格讨论吞吐、显存或
   质量差异，而不是把不同模型/数据的结果混为后端差异。
3. Qwen3.5-9B 是 4B 路径稳定后的下一档；Qwen3.6/3.8 官方 27B 与 MoE 模型
   需要单独的显存和并行设计，不是这份 4B 配置的直接放大。
4. 将已验证的“数据 -> rollout -> reward -> group advantage -> logprob -> GRPO
   update -> metrics”映射回 mini-verl；mini-verl 保持单机教学实现，不复制
   Ray/FSDP/vLLM 的生产编排。

## 2026-08-19：官方单步通过；两步显存边界

官方单步 artifact `qwen3.5-4b-openr1-grpo-one-step-v5-short-20260819T1705` 已以
exit status 0 完成，并保存完整 `global_step_1`。其 reward mean 为 0.375，
advantage 范围 [-1.50, 0.50]，actor loss 为 -0.00498，grad norm 为 6.54，
回答平均 116 token 且没有 response-cap 截断。因此，384-token 短解答契约已在
官方 vLLM/FSDP2 路径中成立。

两步校准另定位了两个独立 upstream 约束：`train_batch_size * rollout.n` 必须能被
agent-loop worker 数均分（启动器现会在模型初始化前检查）；合法拓扑均能完成并保存
第 1 步，但第 2 步 actor update 在 GPU 0 固定额外申请约 1.19 GiB 时 OOM。
参数 offload 和 PyTorch `expandable_segments:True` 已验证不能消除这个 update 峰值。
该边界不是 reward 退化、数据耗尽或 vLLM 故障。

## 当前状态与下一步

已完成：OpenR1 转换、MATH/GSM8K exact de-dup、官方 `math_reward` router
校验、512-token 的显存边界尝试、256-token 的 all-zero reward 反证，以及多轮
冻结权重 rollout 诊断。旧数据和失败 artifact 全部保留。512-token 运行的第 1 个
真实 update/checkpoint 成功，但第 2 个 backward 额外请求约 1.19 GiB 时 OOM；
256-token 运行虽然完成第 1 个 checkpoint，却在第 2 个 actor update 卡住且第 1 步
reward 全零，因此两者都不是有效质量训练。

新 v5 数据固定短解答契约：2,048 行训练、64 行验证以及一个行号为 `0,1` 的两行
校准 slice。官方单步已通过该门槛。接下来不再重复 2 trainer + 2 rollout 的 allocator
微调，而是将四卡划为 **3 张 FSDP2 trainer + 1 张 TP=1 vLLM rollout**。这会降低
每张 trainer 卡的模型、梯度与 optimizer shard 压力，同时仍只使用已空闲的四张卡。

首个 3+1 验证已用从冻结 2,048 题中创建的全新六题、无 shuffle slice：
`train_batch_size=3`、`rollout.n=4`、`ppo_mini_batch_size=3`、4 个 agent worker 和
两步。每步生成 12 条 trajectory，既能被 3 张 trainer 卡整除，也能被 4 个 reward
worker 均分。它从 Qwen 基座重新开始，**没有**加载 2+2 FSDP checkpoint。

artifact `qwen3.5-4b-openr1-grpo-two-step-v5-short-trainer3-rollout1-20260819T1820`
以 `exit_status=0` 完成（总时长 576.66 秒），完整保存 world-size=3 的
`global_step_1` 与 `global_step_2` actor/optimizer shards。TP=1 vLLM 的加载、
weight sync、rollout 和两次 FSDP2 update 都成功，故 3+1 解决了 2+2 拓扑的第二次
update OOM 边界。第 1 步 12 条 rollout 的 reward 为 11 个 0、1 个 1，产生
advantages `[-0.50, 1.50]`、actor loss `-0.02173`、grad norm `4.07`，且没有
response cap 截断；这是一次真实的非退化 GRPO 更新。第 2 步的 12 条 reward 恰好
全为 0，因而 policy-gradient loss 为 0、grad norm 约 `0.0014`，尽管 update/checkpoint
仍完整完成。

因此这条 run 证明**内存与系统稳定性**，也证明至少一个 batch 能产生有效学习信号；
它不证明两步已让模型泛化提升，更不能据此跳到 10k。初始/最终 64 题 MATH-lighteval
accuracy@1 从 0.625 变为 0.65625，中间 step 1 为 0.671875；样本小且步数太少，
仅记录而不解释为质量提升。下一步是在不占用他人 GPU 的前提下，用冻结权重的
rollout audit 从 2,048 题中筛出多个可重复出现组内 reward 差异的 batch；再在相同
3+1 拓扑上做更长但仍小规模的稳定训练，之后才考虑 2,048/10k。

## 2026-08-19：审计筛选后的双步有效更新

该 frozen rollout audit 已完成，随后以原始行号 `0,1,24,25,26,34` 构成六行、
不 shuffle 的训练输入，在同一 3+1 拓扑上重新从 Qwen 基座运行两步。该训练的实际
rollout（不是 audit 的复用结果）两步均有混合 0/1 reward：第 1 步为 4/12 正例，
第 2 步为 7/12 正例；每步均有非零 relative advantage、actor loss 和 grad norm。
运行以 exit status 0 完成，保存两个 world-size=3 checkpoint，GPU 自动释放；artifact
为 `qwen3.5-4b-openr1-grpo-two-step-v5-short-trainer3-rollout1-audited-20260819T1847`。

这解除的是“第二步可能没有学习信号”的门槛，不是质量验证。64 题 MATH-lighteval
accuracy@1 从 0.640625 到 0.671875 仅作记录，不可解释为泛化提升。下一次仍保持模型、
奖励、prompt 契约、384-token cap、3+1 拓扑和安全 preflight 不变，只将固定的审计
行扩展为 15 条 / 5 步（每步 3 prompt × 4 rollout）；若它同样 clean exit 且没有
持续全零 batch，才进入固定 2,048 题的短训设计。

## 2026-08-19：五步门槛通过，进入 2,048 行短训

15 条审计行的五步 run 已 clean exit：五个实际训练 batch 均至少有一个 0/1 混合
reward group，正例数依次为 2、5、2、6、4（各 12 条 rollout），gradient norm 为
7.36、8.82、4.48、4.77、8.72；没有 all-zero batch 或 OOM，完整保存
`global_step_1` 到 `global_step_5`。因此当前契约足以进入固定的 2,048 行一轮短训，
不再重复小切片试验。

正式短训仍不等于“模型已学会”：`train_batch_size=3` 意味着约 683 个 update，按本机
实测约 80--100 秒/update，预估 15--19 小时。为避免每步 checkpoint 约 53 GiB 导致
磁盘耗尽，正式 run 使用稀疏 `SAVE_FREQ` 与 `TEST_FREQ`，保留可恢复中间点、最终
checkpoint、全部 rollout samples、训练日志以及定期的 64 题冻结评测。只有完整 run
退出后，才能比较基线/中间/最终评测和 reward 统计；中途的训练 reward 不作为质量结论。

在实际启动时，原 2,046 行候选被官方 Qwen3.5 `AutoProcessor` prompt-length filter
移除了 8 行，故不能声称它恰为 682 个完整 batch。该任务在任何 rollout 前被安全停止；
最终正式输入严格复现上游 processor 的长度契约，包含 2,037 行 / 679 个完整 batch，
并已有 upstream 启动日志确认。当前正式 artifact 为
`qwen3.5-4b-openr1-grpo-2037row-679step-v5-short-trainer3-rollout1-20260819T1919`，
中点和最终 checkpoint 分别为 step 340 与 679，冻结 64 题评测在 170、340、510、679
步进行。
