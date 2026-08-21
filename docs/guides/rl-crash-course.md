# 强化学习速成（RL/RLHF 精髓）

写给"想尽快懂 RL 精髓"的人。不求数学完备，只求直觉正确、术语对得上。

---

## 1. 一句话抓住 RL

**RL = 让模型自己试，然后用"结果好不好"来调整它。**

对比一下三种学习方式：

| 方式 | 数据 | 信号 | 典型场景 |
|---|---|---|---|
| 监督学习 (SFT) | 输入→**标准答案** | 答案直接告诉你对错 | 预训练、指令微调 |
| 强化学习 (RL) | 输入→**模型自己的输出** | 只有"好不好"的分数 | 让模型学会推理、对齐 |
| 自监督 (预训练) | 只有文本 | 猜下一个 token | GPT 预训练 |

关键区别：**SFT 教模型"照抄"，RL 教模型"探索并自我改进"**。
SFT 需要有人写好答案（上限=人类水平）；RL 只要有个打分函数（reward），模型自己能发现人类没想到的解法。

---

## 2. RL 的核心四要素

```
     ┌─────────────────────────────────────────────┐
     │  1. Policy (策略) = 模型本身                 │
     │     给定问题 → 生成回答的概率分布            │
     │  2. Reward (奖励) = 打分器                   │
     │     回答 → 分数（对错/质量）                 │
     │  3. Environment (环境) = 出题方 + 判卷方      │
     │     给问题、收回答、给分数                    │
     │  4. Update (更新) = 怎么改模型               │
     │     分数高的回答概率↑，分数低的↓             │
     └─────────────────────────────────────────────┘
```

整个 RL 的流派之争，本质就是**第 4 点"怎么更新"的不同做法**。

---

## 3. 两大流派：基于策略梯度 vs 基于偏好

### 流派 A：策略梯度（Policy Gradient）—— "用分数直接推"

核心公式直觉：

```
更新方向 = 期望[ 该动作的分数 × 该动作的概率梯度 ]
```

翻译成人话：**"做过的动作里，分数高的多来点，分数低的少来点。"**

这一派的关键问题是：**分数有噪声、有尺度问题**。所以演化出一堆改进：

### 3.1 REINFORCE（最原始）
- **全称**：REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility（Williams, 1992）。这是个"为了凑首字母缩写"的名字，实际就是**最朴素的策略梯度（vanilla policy gradient）**。
- 做法：采一条轨迹 → 看总分 → 正分往上推，负分往下压
- 缺点：方差巨大（同一题做两次，一次对一次错，梯度方向相反）

#### 3.2 PPO（近端策略优化）—— 工业标准
- **全称**：**Proximal Policy Optimization**（OpenAI, Schulman et al., 2017）
- **“近端”是什么意思？** 每轮更新后的**新策略要靠近旧策略**：对已采样回答的生成概率不能一下子相对变化太大。PPO 用新旧策略的概率比 `ratio = 新概率 / 旧概率`，并用 clip 把有效变化限制在小范围（常见为 `0.8 ~ 1.2`）。这不是说模型参数在几何空间里一定很近，而是说**行为概率别突然变样**，避免被一批有噪声的 reward 带偏。
- 改进 1：**用新旧策略的概率比**，限制单次更新别太大（clip），训练稳定
- 改进 2：**引入 Critic（价值网络）**，估计"这个状态本来该得多少分"，
  用 **Advantage = 实际分数 − 预期分数** 作为更新信号
- 缺点：**要额外训一个 Critic 网络**，显存翻倍、训练不稳

#### 3.3 GRPO（群组相对策略优化）—— 本仓库用的
- **全称**：**Group Relative Policy Optimization**（DeepSeekMath, Shao et al., 2024）
- **去掉 Critic**：让模型对同一道题生成 N 个回答（如 4 个）
- 用这 N 个回答的**组内相对分数**替代 Critic 的"预期分数"
- **Advantage = (自己的分数 − 组平均) / 组标准差**
- **不是把 PPO 全盘换掉**：GRPO 换掉的是"Critic 计算 advantage"这一步；策略更新仍沿用 PPO 风格的"新旧概率比 + clip"，限制一次更新别偏离旧策略太远。
- 好处：省一个网络、更稳、天然适合"对错明确"的任务（数学/代码）

> **为什么 GRPO 适合数学？** 数学题 reward 是"答案对不对"，绝对分数 0/1 太稀疏。
> 组内相对比较让模型看到"同一题我这次答对了/上次答错了"的差异，信号更密。

---

### 流派 B：偏好优化（Preference Optimization）—— "用人类喜好直接学"

没有显式分数，只有**成对比较**："回答 A 比回答 B 好"。

| 算法 | 核心思想 | 一句话 |
|---|---|---|
| **RLHF + PPO** | **RLHF（Reinforcement Learning from Human Feedback）**：先根据人类偏好训练 Reward Model（打分器），再用 **PPO（Proximal Policy Optimization，近端策略优化）** 更新语言模型 | 人类喜好 → 学一个分 → PPO 稳定地让模型更偏向高分回答 |
| **DPO** | **Direct Preference Optimization**（Rafailov et al., 2023）：跳过 Reward Model，直接用偏好对优化 | "这个好那个差"直接指导更新 |
| **KTO** | **Kahneman-Tversky Optimization**：只用"好不好"（单样本），不用成对 | 更省数据 |

**为什么会有 DPO？** RLHF 管线重（要训 RM + PPO 两个阶段）。
DPO 发现：RM + PPO 的组合其实可以闭式解出来，**直接比较两个回答的 log 概率差**就能更新，
不用训 RM，不用 RL 循环。代价：没有探索能力，上限低于好的 RL。

---

## 4. 为什么 RLHF 是"三步走"？

```
第一步 SFT：让模型会说话、会跟随指令（模仿）
第二步 RM：人类给一堆回答打分 → 训一个"自动打分器"（替代人）
第三步 RL：用 RM 当 reward，PPO/GRPO 优化（让模型越答越好）
```

为什么要分三步？
1. **模型得先会说人话**才能 RL（否则探索的都是乱码）
2. **人类打分太慢太贵**，训个 RM 自动打（本质是"把人类偏好编码成模型"）
3. **RL 让模型超越人类示范**——SFT 只能模仿，RL 能发现新解法

---

## 5. 一个具体例子：你的 GRPO 训练在干嘛

```
问题："x² - 7x + c = 0 有实根且有理根，求 c"
  │
  ├─ 模型生成 4 个回答（GRPO 组）：
  │     A: 判别式 D=49-4c 是完全平方 → c=12,10,6  ✓
  │     B: c=12,10（漏了 6）                    ✗
  │     C: ...（跑题）                          ✗
  │     D: c=6,10,12 乱序但对                    ✓
  │
  ├─ reward：A=1, B=0, C=0, D=1
  │
  ├─ GRPO 算组内 advantage：
  │     A: (1−0.5)/σ > 0  ↑ 强化
  │     B: (0−0.5)/σ < 0  ↓ 弱化
  │     C: (0−0.5)/σ < 0  ↓ 弱化
  │     D: (1−0.5)/σ > 0  ↑ 强化
  │
  └─ 更新：模型下次更倾向"像 A/D 那样完整推理"
```

这就是你看到 64 题 accuracy 从 65.6% → 87.5% 的机制。

---

## 6. RL 的常见坑（对应本仓库踩过的）

| 坑 | 现象 | 本仓库实例 |
|---|---|---|
| **reward hacking** | 模型找到打分的漏洞 | 答案格式漂移（分数→小数）骗过精确匹配评分器 |
| **reward 稀疏** | 绝大多数回答得 0 分，学不动 | 数学题 0/1 奖励，靠 GRPO 组内对比缓解 |
| **KL 崩溃** | 模型偏离原始能力 | 用 `kl_loss_coef=0.001` + `low_var_kl` 约束 |
| **长度膨胀** | 模型学会啰嗦（reward 不惩罚） | response_length 138→270+，逼近 384 上限 |
| **过拟合训练集** | 背题不泛化 | 2037 题太小，需扩到 10000 + held-out 验证 |
| **解码死循环** | greedy 生成卡死 | 评测脚本加超时（见 [L20 实验经验](../operations/l20-lessons-learned.md)） |

**RL 的黄金法则：reward 定义了什么，模型就优化什么。**
你 reward 只查"答案对不对"，模型就会钻格式空子、会堆字数；
如果 reward 还惩罚"超出三句话"，模型才会学简洁。

---

## 7. 术语对照速查

| 术语 | 意思 |
|---|---|
| Policy | 模型（给定输入→输出的概率分布） |
| Reward | 打分函数（回答→分数） |
| Advantage | 实际分数 − 预期分数（"这次比平时好多少"） |
| Value/Critic | 预测"预期分数"的网络（PPO 用，GRPO 不用） |
| KL 惩罚 | 限制模型别偏离原始模型太远 |
| Rollout | 模型生成的一批回答（采样） |
| Group Relative | 组内相对比较（GRPO 的 G） |
| Reward Model (RM) | 从人类偏好学出来的打分器（RLHF 用） |
| SFT | 监督微调（模仿） |
| RLHF | 用 RL 对齐人类偏好（RM + RL） |

---

## 8. 如果想深入

1. **Spinning Up in Deep RL**（OpenAI 免费教程）—— 最友好的 RL 入门
2. **PPO 原论文** "Proximal Policy Optimization Algorithms" —— 工业标准
3. **GRPO 出处**：DeepSeekMath / DeepSeek-R1 论文 —— 本仓库算法的来源
4. **DPO 原论文** "Direct Preference Optimization" —— 偏好派代表
5. 本仓库实战：[679-step 结果](../results/qwen3.5-4b-grpo-679-step.md) + [L20 实验经验](../operations/l20-lessons-learned.md)

---

## 9. 面试 Q&A（一问一答速记）

### Q1: GRPO 和 PPO 的区别？
- PPO 需要**一个额外的 Critic 网络**估计"预期分数"，GRPO **不需要**——用同题生成的 N 个回答的**组内相对分数**替代 Critic。
- 两者都要防止策略一次改太猛：都可使用 PPO 风格的**新旧概率比 + clip**。因此 GRPO 的关键不是"不用 PPO 的稳定更新"，而是"不用 PPO 的 Critic 来算 advantage"。
- 代价/收益：GRPO 省一个网络（显存减半、更简单稳定），但只能用于**能批量采样、组内可比**的任务（数学/代码对错明确）。
- PPO 更通用（能处理连续控制、单轨迹场景），GRPO 是"为 LLM 数学/代码 RL 定制"的简化。

### Q2: 什么是 advantage（优势）？
- **advantage = 实际分数 − 预期分数** = "这次比平时好多少"。
- PPO 的"预期分数"来自 Critic；GRPO 的"预期分数"来自**组内平均分**。
- 减掉预期后，只有"超出平时水平"的部分驱动更新，方差小、训练稳。

### Q3: 为什么数学任务用 GRPO？reward 0/1 太稀疏是什么意思？
- 数学 reward 只有 0（错）/1（对），**绝大多数样本是 0**，绝对信号几乎没有方向。
- GRPO 的**组内相对比较**把 0/1 变成**连续的相对分**：同一题 4 个回答，
  advantage = (自己的分 − 组平均) / 组标准差 → 得到 +1.2 / -0.3 / -0.9 这种连续值。
- 即使**全组都错**（4 个 0 分），相对比较仍能区分"接近对的思路"vs"完全跑题"，
  梯度方向连续、每步都有信息量 → "信号更密"。

### Q4: kl_coef=0.001 + low_var_kl 是什么意思？
- KL 惩罚 = 约束"训练后的模型"别偏离"训练前的参考模型"太远（防丢失预训练能力）。
- `kl_coef=0.001`：`total_loss = policy_loss + 0.001 × KL`，小权重 = 允许改变但别放飞。
- `low_var_kl`：KL 的一种低方差估计形式 `exp(ref−new) − (ref−new) − 1`，
  比朴素估计更稳（本仓库 `grpo.py` 用的正是这个公式）。
- 训练中观测：ppo_kl 稳定在 0.011–0.024，说明约束生效、未发散。

### Q5: 什么是 trajectory / rollout？
- **rollout** = 模型针对一道题生成的一批回答（采样产物）。
- **trajectory** = 一条完整记录：prompt + response + old_logprobs + reward + policy_version。
- 训练时 rollout 采样，trajectory 运输，reward 打分，GRPO 算 advantage，再更新。

### Q6: 什么是 one-step policy lag / rollout 准入？
- **本仓库的正式 4B verl run**：是“训练当前 batch 时，异步生成下一 batch”的
  **one-step-overlap 受控异步流水线**，不是全同步的 rollout → train → rollout 串行循环。
  训练和下一批 rollout 重叠，但下一批最多滞后一代 policy，因此不是无约束异步。
- **policy lag**：允许"用旧策略 v_k 采样的 rollout"在训练推进到 v_{k+1} 后再被消费，
  换取 rollout 与训练**流水线重叠**（不等生成，吞吐↑）。`max_policy_lag=1` 即允许滞后一步。
- **rollout 准入**：消费前检查轨迹的 policy_version，发现**过期/未知版本直接拒绝**，
  防止用错策略的样本训练。

### Q6b: 同步与异步训练各有什么好处？

**同步**是严格串行：`生成 batch k → 打分 → 训练 batch k → 同步权重 → 生成 batch k+1`。
它的优点是样本一定来自最新策略，语义简单、容易调试和复现；缺点是 rollout GPU 与 trainer
GPU 会轮流空闲，单轮耗时接近“生成时间 + 训练时间”。

**异步**让 trainer 训练 batch k 的同时，rollout GPU 生成 batch k+1。优点是两类 GPU
可以重叠工作，单轮耗时更接近两者中较慢的一项，吞吐更高；代价是下一批样本可能由旧策略
生成。若样本滞后很多代，当前策略与采样策略相差过大，更新可能不稳，同时还要处理版本检查、
权重同步和故障恢复。

因此本项目选择**最多一代滞后的受控异步**：比全同步更充分利用 GPU，又用
`max_policy_lag=1` 限制旧样本，避免无约束异步的陈旧样本风险。

### Q7: 你的实验里 KL、advantage、length 各说明了什么？
- ppo_kl 0.011–0.024 稳定 → 模型没偏离原始能力（KL 约束生效）。
- advantage 均值在 0 附近摆动 → 组内归一化正常（有正有负才正常）。
- response_length 138→270+ → 模型在"越写越长"（reward 不惩罚啰嗦，是下一轮要修的坑）。

---

## 10. RL 训练框架对比（VeRL / SGLang / Swift / AgentRL）

> 这一节回应简历上的"熟悉 VeRL、Slime、Swift 等框架"。
> 说明：Slime 是 SGLang 生态里 agentic RL 的框架（sglang 团队），
> 下面对比时以它为准；如果指别的同名项目，以实际为准。

### 10.1 四个框架一句话定位

| 框架 | 团队 | 一句话定位 | 适用任务 |
|---|---|---|---|
| **VeRL** | Bytedance（火山引擎） | 字节开源的 LLM RL 框架，最主流、最完整 | 单轮 RL（数学/代码/通用 RLHF） |
| **SGLang (Slime)** | LMSYS / sglang 团队 | 高性能推理引擎 + 新出的 agentic RL 框架 | agent RL（多轮工具调用） |
| **Swift** | 阿里（魔搭 ModelScope） | 一站式 LLM 训练/微调/RL 框架，开箱即用 | RLHF/DPO/GRPO + SFT，偏易用 |
| **AgentRL** | 清华 + Z.AI | 多轮多任务 agentic RL 框架（AutoGLM 背后） | agent RL（网页/手机/工具） |

### 10.2 详细对比

**VeRL（你实际用的）**
- 字节开源，PyTorch 生态，支持 FSDP/FSDP2 + vLLM 做 rollout
- 特点：单轮 RL 事实标准，数学/代码 RL 大量用它（DeepSeek 系复现的首选）
- 你的实验用的就是它：`verl.experimental.one_step_off_policy.main_ppo` + GRPO
- 局限：多轮 agent 支持相对弱（1.x 主攻单轮；agent 能力在实验版/社区分支）

**SGLang / Slime**
- SGLang 是推理引擎（RadixAttention 前缀缓存、连续批），比 vLLM 新、快
- Slime 是它的 agentic RL 框架：专门做多轮工具调用 RL（浏览器、代码执行）
- 特点：多轮 rollout 的调度、轨迹存储、环境交互都是原生设计

**Swift**
- 阿里的"全家桶"：数据准备 → SFT → RLHF/DPO/GRPO → 评测，一条龙
- 特点：**易用性第一**，配置化、文档全、社区大，适合快速上手/业务落地
- 性能/灵活性不如 VeRL，但胜在省事

**AgentRL（THUDM）**
- 清华 + Z.AI，AutoGLM 背后的框架，2025-10 开源
- 特点：多轮多任务，全异步生成-训练流水线，跨策略采样 + 任务级 advantage 归一化
- 定位：学术前沿 + 生产 agent 训练

### 10.3 选型速查

| 你的需求 | 选谁 |
|---|---|
| 单轮数学/代码 RL，要复现论文 | **VeRL** |
| 快速上手 RLHF/DPO，不想折腾 | **Swift** |
| 多轮 agent RL（工具/浏览器） | **SGLang/Slime** 或 **AgentRL** |
| 多任务 agent RL + 研究前沿 | **AgentRL** |

### 10.4 和你的关系

- 你已经**精通 VeRL**（679 步实战 + 踩坑）
- 简历"熟悉 PPO/DPO/GRPO"——算法层已覆盖（本文档）
- 想补"熟悉 Slime/Swift"：最快路径是拿同一份数据各跑一个 smoke
  （Swift 有现成 GRPO 示例；Slime 用它的 agent 示例），半天能出对比结论

---

## 11. 模型架构：Qwen3.5-4B vs Qwen3-0.6B（本仓库用过的两个模型）

> 面试被问"你用的模型什么架构"时的速答版。
> 数据来自 L20 上两个模型的实际 config.json。

### 11.1 一表对比

| 维度 | **Qwen3.5-4B**（GRPO 训练用） | **Qwen3-0.6B-Base**（smoke 用） |
|---|---|---|
| model_type | qwen3_5（多模态生成） | qwen3（纯文本因果 LM） |
| 层数 | 32 | 28 |
| hidden_size | 2560 | 1024 |
| FFN（SwiGLU） | 9216 | 3072 |
| 注意力头 / KV 头 | 16 / **4**（GQA） | 16 / **8**（GQA） |
| head_dim | 256 | 128 |
| 上下文长度 | **262144（256K）** | 32768（32K） |
| 注意力类型 | **混合：24 linear + 8 full**（每 4 层 1 full） | 28 层全 full attention |
| attn_output_gate | ✅ 有 | ❌ 无 |
| vocab | 248320（含图像/视频 token） | 151936 |
| tie_word_embeddings | ✅ | ✅ |
| 实际参数 | ~5.0B | ~0.8B |

### 11.2 核心差异：混合注意力（hybrid linear + full attention）

**Qwen3.5-4B 的最大亮点**是注意力架构：

- 32 层里 **24 层 linear_attention（线性注意力）+ 8 层 full_attention（标准注意力）**，
  按 `full_attention_interval: 4` 每 4 层插一个 full 层（pattern: L L L F L L L F ...）。
- **线性注意力**把自注意力从 O(n²) 降到 O(n)，是支持 256K 超长上下文的根本原因。
- **full 层做"锚点"**：每隔几层用标准注意力做全局信息校正，弥补线性注意力的信息丢失。
- `attn_output_gate`（输出门控）：对注意力输出再做一个门控，增强表达能力。

这是当前业界"长上下文 + 高效"的主流方向：
- DeepSeek-V3 的 DSA（DeepSeek Sparse Attention）也是"稀疏/混合注意力"路线；
- Qwen3.5 用 24:8 的混合比例，比纯线性（如 Mamba 系）保留更好的全局建模，比全注意力省显存/算力。

**面试可讲的一句话**：
> "Qwen3.5-4B 是混合注意力架构，32 层里 24 层线性注意力 + 8 层全注意力（每 4 层一个 full），
> 线性注意力把复杂度从 O(n²) 降到 O(n)，所以能支持 256K 上下文；full 层作为全局锚点弥补信息丢失，
> 类似 DeepSeek-V3 DSA 的思路。"

### 11.3 GQA（分组查询注意力）——两个模型都有

- **GQA = Grouped Query Attention**：多个 query 头共享一组 KV 头。
  - Qwen3.5-4B：16 query / 4 KV（4:1 分组）
  - Qwen3-0.6B：16 query / 8 KV（2:1 分组）
- **为什么用 GQA**：KV cache 显存随 KV 头数量线性增长，GQA 砍掉大部分 KV 头，
  推理显存和带宽都省，效果损失很小（业界标准，Llama 3 / Mistral 都用）。
- 面试："我的模型用 GQA，4B 是 16 query 共享 4 个 KV 头，KV cache 省 4 倍。"

### 11.4 参数口径：4B 不是 4B

- Qwen3.5-4B 实际 ~**5.0B** 参数（含 embedding），命名是营销口径。
- Qwen3-0.6B 实际 ~**0.8B**。
- 且都 `tie_word_embeddings=True`（输入/输出 embedding 共享），省一大块参数。
- 面试：主动提"叫 4B 实际 5.0B，因为 tie embedding"显专业。

### 11.5 多模态 vs 纯文本

- Qwen3.5-4B 是 `Qwen3_5ForConditionalGeneration`，config 里有
  `image_token_id` / `video_token_id` / `vision_start/end_token_id`——**支持图像和视频输入**。
  你 GRPO 只用它的文本分支，但模型本身是多模态。
- Qwen3-0.6B 是 `Qwen3ForCausalLM`，纯文本。

### 11.6 和 RL 实验的关系（面试可能追问）

- Qwen3.5-4B 训练时 `max_response_length=384` token，**远没吃满 256K 上下文**——长上下文优势在短回答场景用不上。
- 但混合注意力在 4B 级别提供了强推理基础，且 GRPO 下收敛稳定（你的 679 步实验验证了这点）。
- 如果面试官问"为什么不用更大的模型"：L20 是 48GB×4 卡，4B + FSDP2 + vLLM 是显存和吞吐的平衡点；
  更大模型（7B+）单卡放不下，多卡通信又受 PCIe 限制（L20 无 NVLink）。
