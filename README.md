# mini-verl

从零实现、面向理解与性能实验的 **LLM 强化学习训练框架**。项目以 `verl` 的核心数据流为原型，但不追求 API 兼容；目标是用可读、可测、可 profile 的最小实现，打通 rollout、奖励、GRPO 更新和分布式资源协同。

## 为什么做它

LLM 后训练的强化学习不是单个 loss 函数：同一轮训练同时涉及生成式推理（rollout）、奖励计算、轨迹数据、策略版本管理、显存复用和分布式更新。`mini-verl` 将这些系统边界显式化，并用端到端实验验证其正确性和瓶颈。

```text
prompt batch
    |
    v
RolloutWorker (actor policy v_k) -- trajectories --> RewardWorker
    ^                                                   |
    |                                             rewards / advantages
    |                                                   v
    +---- policy sync <--- TrainerWorker (GRPO update)
                                  |
                              policy v_(k+1)
```

## 三分钟跑通

```bash
python -m pip install -e '.[torch]'
mini-verl-toy
# 或：python -m mini_verl.toy
```

这条 CPU、无模型下载的最小训练会打印从 `initial_pass@1=0.125` 到
`final_pass@1=1.000` 的可验证提升。核心代码在
[`mini_verl/toy.py`](mini_verl/toy.py)：它只替换了语言模型本身，保留了
verl 最关键的 rollout → old logprob → reward → group advantage → clipped
GRPO update → policy version 数据流。阅读顺序见 [CORE.md](CORE.md)。

## v0 范围与非目标

**首个可交付版本：**一个无需下载模型、可在 CPU 上运行的最小 GRPO 闭环，以及同一数据契约下的 Hugging Face 单机/多卡参考 backend：规则奖励、reference-KL、策略版本同步和可复现的格式验证任务。

**刻意不做：**完整 verl API 兼容、Ray 集群编排、多模态、任意模型/奖励模型兼容、生产级容错；这些会在核心闭环已正确且有 benchmark 的前提下再逐步扩展。

## 计划

### Phase 0 — 设计与基础设施

- [x] 定义 `Trajectory` / `TrajectoryBatch` 协议：`prompt_ids`、`response_ids`、`response_mask`、`old_logprobs`、`ref_logprobs`、`reward`、`advantage`、`policy_version` 和文本/元数据。
- [x] 约定 `RunConfig`、统一随机种子、结构化 iteration metrics 和 checkpoint 格式。
- [x] 准备可验证的算术/格式任务与 rule-based reward，建立无模型依赖的 smoke test。

**验收：**一条 rollout 在序列化/拼接后 token、mask、logprob 和 policy version 不变；固定 seed 下 smoke test 可复现。

### Phase 1 — 正确的 GRPO 闭环

- [x] 实现可验证的 rollout：每个 prompt 采样 `G` 条 response，并记录旧策略 token logprob（toy categorical policy + Hugging Face causal-LM backend）。
- [x] 实现 reward、组内标准化 advantage、response mask 和 GRPO clipped objective（含 dependency-free reference 与 PyTorch 后端）。
- [x] 实现 reference policy KL 项与单卡 trainer（包含可微 PyTorch loss、causal-LM response logprob 对齐、Hugging Face backend 与端到端 toy GRPO loop）。
- [x] 输出 reward、response length、KL、clip fraction、rollout/train token throughput 与 stage timings。

**验收：**

- policy 与 old policy 相同时，importance ratio 为 1；
- reward 做组内整体平移时 advantage 不变；
- padding token 不参与 loss；
- 在固定的小任务上，训练后平均 reward 和 pass@1 相对初始策略可重复提升。

当前已完成：依赖无模型下载的 toy GRPO 闭环，以及 Controller 驱动的 Hugging Face CausalLM rollout/old-logprob/trainer 集成路径。最短入口是 `python -m mini_verl.toy`；它在一个小型 categorical policy 上真实执行 rollout、规则奖励、组内 advantage、clipped GRPO update 与 policy version 推进。通用的本地 Hugging Face 模型 smoke 位于 `examples/hf_grpo_smoke.py`，不依赖官方 verl。

### PPO actor--critic 教学对照

除 GRPO 主线外，项目还提供一个刻意很小的 PPO actor--critic 对照：
`examples/toy_ppo_train.py` 和 `mini_verl/algorithms/ppo.py`。它与 toy GRPO 使用相同的
16-prompt categorical 环境、`0/1` rule reward、old logprob、PPO-style ratio/clip 和
`pass@1` 验收；差别只在 advantage 的来源：PPO 用 Critic + GAE，并额外最小化 value loss，
GRPO 用同题回答的组内相对 reward。

这使“Critic 为什么存在、PPO 与 GRPO 哪部分相同”可由代码和数值单测检验；它不是 LLM
多 token PPO、不是分布式 Critic，也不构成任何官方 verl 或 4B 训练结论。

### Phase 2 — 框架化与分布式训练

- [x] 拆分 `RolloutWorker`、`RewardWorker`、`TrainerWorker`、`PolicySynchronizer` 和 `Controller` 的生命周期与数据契约。
- [x] 接入最小 DDP 运行时，并用双卡 GRPO smoke 验证不同 rank 的梯度平均后策略副本同步；ZeRO 后端待实现。
- [x] 实现单进程 checkpoint、恢复、RNG 状态与 policy version 持久化，以及 trainer→rollout 的全量 state-dict 同步；分片权重同步待实现。
- [x] 处理变长 sequence 的 length bucketing / padded-token 预算，并增加双卡 DDP 参数一致性测试。

**验收：**多卡与单卡在可比设置下 loss/参数满足数值容差；恢复训练轨迹一致；过期 trajectory 能由策略版本检查拒绝或被显式标记。

### Phase 3 — 训推协同与性能实验

- [x] 抽象 rollout backend：`RolloutWorker` 协议已支持 HF generate；vLLM/SGLang backend 不是 main 的 v0 范围。
- [x] 实现阶段式同步 Controller，以及 one-step policy-lag 的 PrefetchingController：learner 优化当前 batch 时，用独立 rollout 副本预取同版本的后继 batch；待生成完成后才同步副本到下一版本，后继 batch 以最多一代 lag 被消费。资源分离的多进程/多节点流水线待实现。
- [x] 记录 rollout（含可选 generate prefill/decode/old-logprob forward）、reward、训练、权重同步阶段耗时、token 吞吐、PyTorch allocator 峰值显存与 opt-in 的 `nvidia-smi` 设备级利用率采样。
- [x] 增加离线构造 tiny GPT-2 的 Hugging Face 端到端阶段基准：实际执行 `generate`、old-logprob 回算和 CausalLM GRPO 更新，分别报告 rollout/reward/train；仍不等同于真实业务模型。
- [x] 实现 correctness-first 的 HF prompt micro-batching（左填充、trajectory 契约不变）：在相同 tiny GPT-2 任务上把 `rollout_batch_size=1` 提升到 4，使 rollout `66.807 → 19.487 ms`、端到端 iteration 2.78×。
- [x] 在静态 rollout micro-batch 上实现可选 prompt-length bucketing：先编码、按长度组批、生成后还原原始 prompt/sample 顺序，并显式统计 group 扩展后的 prompt padding。交错长度 `3/24/4/23/5/22/6/21` 的受控 L20 对照将 prompt padding 从 41.3% 降至 10.0%，但 tiny GPT-2 rollout `40.775 → 40.740 ms` 基本持平；该负结果表明此工作负载主要受 decode / HF 固定开销而非 prompt padding 限制。
- [x] 为 rollout static batch 增加 `rollout_max_padded_prompt_tokens` 容量保护：以 `group_size × batch rows × batch max prompt length` 限制每次 `generate` 的扩展后 prefill tensor，单条超限 prompt 显式拒绝；同时报告每轮累计与单 batch 峰值的 padded prompt token。它是用于显存/上下文准入的控制面，而非 KV cache 全量估算。
- [x] 增加可选 `rollout_max_padded_sequence_tokens`：按 `group_size × batch rows × (batch max prompt length + max_new_tokens)` 保留 worst-case decode/KV sequence 容量。它可单独或与 prefill budget 组合；L20 实测以 240 上限将单 batch sequence token 峰值从无预算的 512 限为 232，并在模型调用前拒绝单请求超限。
- [x] 在 HF rollout 前做静态 context-window guard：当模型暴露 `max_position_embeddings` 或 GPT-2 `n_positions` 时，强制 `prompt_tokens + max_new_tokens` 不超过该容量；不可能完成的固定长度请求在 `generate` 前显式报错，而不是依赖后端深层异常。
- [x] 将变长 prompt、length bucketing 与 prefill/sequence admission 接入真实 `HF rollout → old-logprob → reward → GRPO update` 基准。L20 的 8 条交错长度 prompt 上，`sequence budget=240` 将单 batch worst-case sequence token 峰值 `512 → 232`，但因 1→3 次 generate 使 iteration `32.487 → 66.286 ms`；容量保护与吞吐优化必须作为显式取舍报告。
- [x] 将相同 admission 工作负载接入双 GPU trainer/rollout 同步与 one-step-lag prefetch benchmark，并从真实 rollout 输出 admission 指标。`sequence budget=240`、峰值 `232/240` 下，prefetch 将稳态 iteration `68.319 → 62.346 ms`（-8.7%）；受限 rollout 仍有 46.060 ms wait tail，说明第二张 GPU 只能隐藏训练窗口内约 14.9 ms 的生成。
- [x] 将 trainer-side length bucketing 落到真实 GRPO update：多个长度微批按有效 response-token 加权累积梯度、仍只做一次 optimizer step；CUDA 数值回归确认其与同一整批 SGD update 一致。端到端上将训练 padding `33.69% → 3.69%`、trainer allocator 峰值 `27.59 → 21.02 MiB`，但单卡 train `9.010 → 21.616 ms`；双卡 prefetch 可把更长训练窗口的 33.6 ms 与 rollout 重叠，最终仍应按延迟/显存目标选择。
- [x] 增加双 GPU trainer/rollout 副本的同步 vs one-step-lag prefetch 基准：逐 prompt rollout 时测得 11.4 ms overlap、稳态 iteration `72.928 → 69.000 ms`；micro-batch 后仍可隐藏 11.4 ms，iteration `27.444 → 23.096 ms`。输出完整 rollout、未隐藏等待、有效重叠和 cross-device 全量权重同步耗时。
- [x] 已固定 tiny GPT-2 的真实 HF `generate + old-logprob + GRPO` 基准，完成 `group_size=2/4`、`rollout_batch_size=1/4`、length-bucketing 对照：micro-batching 将 response token 吞吐从 852.8 提升到 2355.0 tok/s；双 GPU prefetch 在生成主导 workload 仅改善 5.4%；length bucketing 将 padding 从 42.5% 降至 0、real-token 吞吐提升 6.0%。vLLM/SGLang 属于另一个 serving 项目，而非 mini-verl v0。

**验收：**每一组优化都有固定模型、prompt、warmup、硬件、序列长度分布和统计口径；报告端到端 iteration time 与每个阶段的归因，而不是只报告单点吞吐。

### v0 完成边界

`main` 的核心 GRPO 闭环已经完成：`python -m mini_verl.toy` 是最短、无下载的可执行验收；Hugging Face、DDP、异步 rollout 和 bucketing 是在同一协议上增加的参考实现。另有一份单 token categorical 的 PPO/Critic/GAE 教学对照，边界见上节。LLM 多 token PPO、真实 value head、分布式 Critic、reward model、Agent 工具轨迹、多机容错和 serving backend 仍是独立后续项目，不应以“补全 mini-verl”为由混入此主线。

## 建议目录

```text
mini-verl/
├── README.md
├── mini_verl/
│   ├── protocol.py       # trajectory 数据契约、版本、校验
│   ├── toy.py            # 最小可执行 GRPO 闭环
│   ├── hf.py             # Hugging Face rollout / trainer backend
│   ├── reward.py         # rule / model reward
│   ├── algorithms/grpo.py
│   ├── algorithms/ppo.py # 单 token actor--critic 教学对照
│   ├── workers.py
│   ├── controller.py
│   ├── distributed.py
│   ├── checkpoint.py
│   ├── batching.py
│   ├── policy_sync.py
│   └── metrics.py
├── examples/
├── benchmarks/
└── tests/
```

## 设计原则

1. **先闭环，后扩展。** 先在小而可验证的任务上证明 RL 真在学习。
2. **协议优先。** trajectory 是 rollout、reward 和 trainer 间的稳定边界。
3. **正确性与性能分离。** 单测证明数学和分布式语义；benchmark 证明优化收益。
4. **按 token 衡量。** LLM RL 的 response 变长，样本/s 往往会误导。
5. **每项性能结论可复现。** 记录模型、权重、输入长度、batch、warmup、硬件与统计方式。

## 已验证的运行结果

- 在 L20 单卡 PyTorch/Transformers 环境中，73 项测试全部通过：包括纯数学 reference、CUDA GRPO 梯度、causal-LM response 对齐、Hugging Face rollout/old-logprob/trainer、左填充 prompt micro-batching、prompt-length bucketing、prefill/sequence token budget、context-window fail-fast guard（稳定输出顺序、group 归属、old-logprob 对齐、padding/峰值统计）、checkpoint、策略同步，以及同步/one-step-lag Controller（含真实双 CausalLM 副本）端到端迭代与 length-bucketing 对照。
- toy GRPO 闭环的 greedy pass@1 从 `0.125` 提升到 `1.000`；末轮同步 iteration 约 `8.7 ms`。100 iteration、1 次 warmup、3 次重复的单卡 L20 中位性能为 `114.2 iteration/s`。该 toy workload 只用于正确性，不代表真实模型吞吐。
- 在两张空闲 L20 上运行 DDP GRPO smoke，两个 rank 使用不同局部轨迹完成梯度平均，最终确认 toy policy 副本同步；另有真实微型 Hugging Face CausalLM 的双卡 smoke。

完整复现命令和口径见 [RUNBOOK.md](RUNBOOK.md)；最近一次 L20 toy CUDA 基准的环境、命令、原始 JSON 与解释见 [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)。

## 上游 verl 对照

`main` 只维护本项目的最小实现、测试与本地性能记录。官方
[`verl`](https://github.com/volcengine/verl) 的锁定环境、Qwen/GSM8K smoke、运行
工件与源码映射属于 `official-verl-grpo` 开发分支；它们与这里的教学实现刻意
分离，避免把一次特定的多卡系统实验误当作 mini-verl 的功能或性能结论。

## 最小面试表述

> mini-verl 是一个面向 LLM 后训练的最小分布式 GRPO 框架。我将 rollout、奖励、trajectory 协议、策略版本同步和训练后端拆开，先验证策略优化正确性，再量化 rollout decode、训练和权重同步在端到端 iteration 中的瓶颈；并以一个小型 PPO actor--critic 对照说明 Critic/GAE 与 GRPO 组内 advantage 的差别。
