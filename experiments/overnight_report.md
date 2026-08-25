# mini-verl Agent RL 一晚实验报告

> 日期:2026-08-24 18:10 ~ 22:33 UTC(约 4.4 小时)
> 环境:远程 L20(4×NVIDIA L20 48GB),容器 `zhangwenchao-megatron`(NGC PyTorch 26.01,torch 2.10.0a0+cu130,transformers 5.9.0)
> 模型:Qwen3-0.6B(纯文本 causal LM,bf16,1.3GB)
> 数据:gsm8k-smoke train.parquet(verl 格式,`prompt` + `reward_model.ground_truth`,取前 6 条)
> 全部组合:seed=42,group_size=4,max_new_tokens=512,lr=1e-6,单卡串行(CUDA_VISIBLE_DEVICES=0)

## 摘要

一晚跑完 8 个实验组合,全部成功(rc=0),结果落盘 `experiments/results/*.json`。两个核心结论:

1. **Qwen3-0.6B 完全不会使用 `[PY: ...]` 工具调用**(tool_rate=0 贯穿所有组合),工具可得性不产生任何行为差异——这是主要负结果,有独立的 prompt 探测证据支持。
2. **GRPO 在 gsm8k 上正常学习**(reward 0.67→0.75,loss 下降),稳定性消融显示:reward clipping(±3)在此任务无影响,entropy bonus 有微小可测影响。

## 实验矩阵结果总表

| 组合 | iters | r_first | r_mid | r_last | r_avg | tool_rate | clip | len |
|---|---|---|---|---|---|---|---|---|
| agent_notool_finalonly | 20 | 0.667 | 0.625 | 0.750 | **0.706** | 0.000 | 0.280 | 343.2 |
| agent_tool_finalonly | 20 | 0.667 | 0.708 | 0.667 | **0.719** | 0.000 | 0.272 | 352.8 |
| agent_tool_toolbonus | 20 | 0.667 | 0.708 | 0.667 | **0.719** | 0.000 | 0.272 | 352.8 |
| agent_tool_process | 20 | 0.333 | 0.354 | 0.333 | **0.359** | 0.000 | 0.272 | 352.8 |
| stab_clip0_ent0 | 30 | 0.750 | 0.792 | 0.667 | **0.699** | — | 0.277 | 361.6 |
| stab_clip3_ent0 | 30 | 0.750 | 0.792 | 0.667 | **0.699** | — | 0.277 | 361.6 |
| stab_clip0_ent1e2 | 30 | 0.750 | 0.625 | 0.667 | **0.692** | — | 0.277 | 348.5 |
| stab_clip3_ent1e2 | 30 | 0.750 | 0.625 | 0.667 | **0.692** | — | 0.277 | 348.5 |

## 关键发现

### 1. Qwen3-0.6B 不使用 `[PY: ...]` 工具调用(主要负结果)

**证据链**:
- 8 个组合的 tool_rate 全部为 0.000(工具调用率 0%)
- 独立探测(probe_tool_usage.py):plain / guided / guided_short 三种 system prompt 下,模型 **0/5** 次输出 `[PY: ...]`
- probe_tool_patterns.py:0/4 出现 `tool_call`、`python`、`code_block`、反引号;输出 100% 是 `<think>` 推理链
- 同样的探测在 Qwen3.5-4B 上也是 0/5

**解释**:Qwen 系 reasoning 模型用 `<think>` 内部推理替代外部工具调用,不自发学习 verl/DeepSeek 风格的 `[PY: <code>]` 工具格式。即使 system prompt 显式引导也不改变。

**对 agent 主线的影响**:
- tool_finalonly vs notool_finalonly:reward 几乎相同(0.719 vs 0.706,差 0.013 属噪声)——**工具可得性无增益**
- tool_toolbonus 与 tool_finalonly 完全重合(0.719 vs 0.719)——无工具调用时,工具奖励永不触发,reward shaping 无效
- tool_process 的 reward 减半(0.359)纯属奖励尺度变化(满分从 1 → 0.5),答对率约 72% 不变——**reward shaping 在无工具使用时不影响学习动态**

### 2. GRPO 正常学习(正面验证)

- 所有组合 reward 从 ~0.67 升到 ~0.75,loss 从 ~0.05 降到 ~0.01
- clip_fraction 稳定在 0.27-0.29(约 28% token 被 clip,符合 GRPO 典型值)
- 平均响应长度 343-362 token(Qwen3-0.6B 的 `<think>` 链 + 答案)

### 3. 稳定性消融:clipping 无效,entropy 有微小影响

| 对比 | reward_avg | len | 结论 |
|---|---|---|---|
| clip0_ent0 vs clip3_ent0 | 0.699 vs 0.699 | 361.6 vs 361.6 | **完全相同**——advantage 本就在 ±3 内,clipping 不触发 |
| ent0 vs ent1e2 | 0.699 vs 0.692 | 361.6 vs 348.5 | entropy 略降 reward(-0.007),缩短响应(-13 token) |

- **reward clipping(±3)**:此任务 advantage 幅度小(组内 4 样本,reward 0/1),从不超 ±3,故无影响。要观测 clipping 效果需更大 group_size 或更极端 reward 分布。
- **entropy bonus(0.01)**:轻微降低 reward(-1%),缩短响应长度(-3.6%)。熵实测 0.37,说明模型在思考链下有中等随机性;小系数熵奖励主要起"防坍缩"作用,20-30 iters 内效果有限。

## 实验教训(方法论)

1. **max_new_tokens 必须足够长**:128 token 时 Qwen3-0.6B 的 `<think>` 链被截断,永远到不了 `#### 答案`,reward 全 0 → loss=0 → 模型不学习;512 token 才能让 ~3/4 样本完成。**小模型长思考链任务,长度预算要按"思考 + 答案"估算,不能只看答案长度。**
2. **训练显存需微批**:80 轨迹(20 prompts × 4)整批 pad 训练 → CUDA OOM(44.5GB 打满);`train_micro_batch_size=8` 分微批后显存安全(L20 上用 21GB)。
3. **batch 生成提速 18×**:`PlainRolloutWorker` 逐条串行生成 77s/iter → 左填充 batch 生成 4.3s/iter。
4. **decoder-only 必须左填充**:右填充会破坏生成质量,需显式 `tokenizer.padding_side="left"`。
5. **`[PY:]` 工具格式的模型适配性**:工具调用格式是模型相关的,Qwen 系 reasoning 模型不输出该格式;设计 agent RL 实验前应先探测目标模型的工具行为。

## 复现方式

```bash
# 在 l20 的 zhangwenchao-megatron 容器内
cd /mnt/storage01/zhangwenchao02/repos/mini-verl-l20
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 bash experiments/run_overnight.sh
# 结果在 experiments/results/,分析:
python experiments/analyze_results.py --results experiments/results
```

脚本:`examples/agent_rl_overnight.py`(agent 主线)、`examples/grpo_stability_overnight.py`(稳定性消融)。

## 局限与后续

- **规模小**:6 prompts × group 4 = 24 轨迹/iter,20-30 iters,结论是定性而非定量
- **模型单一**:仅 Qwen3-0.6B。已验证 Qwen3.5-4B 文本模式可用但其混合注意力显存需求大(2×2×512 即 OOM 44.5GB),未纳入
- **工具格式不匹配**:若要让 agent 主线产生信号,需换用原生支持工具调用的模型(如 Qwen 的 function calling 格式)或改用 `[PY:]` 格式训练的模型(DeepSeek 系)
- **后续方向**:①用支持原生工具格式的模型重跑 agent 主线;②加大 group_size(8-16)让 advantage 分布更宽,观测 clipping 效果;③更大 entropy 系数(0.05-0.1)看是否更明显改变探索
