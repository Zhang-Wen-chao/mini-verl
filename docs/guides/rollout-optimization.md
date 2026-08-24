# Rollout 优化:RL 里为什么生成最慢,以及怎么提速

> 定位:这是 `rl-crash-course.md` 的工程侧续篇。算法篇讲"怎么更新",这里讲"怎么让生成(rollout)更快"。
> 适用场景:面试被问"RL 训练慢在哪""投机解码/量化/PD 分离是什么"、以及给自己仓库选 rollout 优化方向。

---

## 1. 为什么先聊 rollout:它是 RL 循环的绝对瓶颈

RL 一个 iteration 的典型时间构成(本仓库 `docs/engineering/mini-verl-performance.md` 实测):

| 阶段 | 占比 | 说明 |
| --- | ---: | --- |
| **rollout(生成)** | **~90%** | 模型逐 token 采样;小模型上还含 old_logprobs 二次前向 |
| train(更新) | ~9% | GRPO 前向 + 反向 |
| reward(打分) | ~0.3% | 规则打分,几乎不占时间 |
| 其他(记账/同步) | 剩余 | Controller 开销 |

**结论:优化 RL 训练,先优化 rollout;优化 rollout,先理解它内部的两次 GPU 访问模式。**

## 2. 核心概念:prefill 与 decode 是两种完全不同的负载

一次 LLM 推理/采样,严格分两个阶段:

| | **prefill(预填)** | **decode(解码)** |
| --- | --- | --- |
| 做什么 | 一次处理**整个 prompt** | **逐 token** 生成(每步 1 个 token) |
| 计算形态 | 大矩阵乘(长序列 × 权重) | 小矩阵乘(1 个 token × 权重) |
| 瓶颈 | **compute-bound(算力密集)** | **memory-bound(带宽密集)** |
| 关键指标 | TTFT(首 token 延迟) | TPS / ITL(每 token 延迟) |
| GPU 状态 | 算力吃满 | **算力闲置,读权重等带宽** |

**为什么 decode 是 memory-bound?** 每生成 1 个 token,都要把模型**全部权重从 HBM 读一遍**(几 GB),但只做一次 1×N 的小矩阵乘。读权重的时间 ≫ 计算时间,GPU 的 FLOPS 在空转,瓶颈在内存带宽。batch 小时尤其严重;batch 大了计算才逐渐吃满。

**为什么 prefill 是 compute-bound?** 一次处理几千个 token,矩阵乘的规模足够大,算力是瓶颈;带宽反而不是。

> 这两句话有权威背书,可放心讲:
> - "Prefill is a compute-bound process that determines TTFT, while decode is a memory-bound process that determines TPS."
> - "The bottleneck on decode is memory bandwidth, with compute sitting idle at low-to-moderate batch sizes."
> (来源:Philip Kiely《Inference Engineering》/ Pragmatic Engineer 深潜访谈)

**所有 rollout 优化手段,本质上都是在解决"decode 太慢、GPU 太闲"这一件事。**

## 3. 优化手段一:连续批处理(continuous batching)—— 让 GPU 每步都在算

**问题:** 传统静态批处理要等整批生成完才收新请求,批内先完成的请求空等 GPU。

**做法:** 调度器每步把 WAITING 请求按预算收进 RUNNING 批;谁到 `max_new_tokens` 谁立刻走、KV 归还,新请求随即补进来。GPU 每步都在算 token,空闲窗口被填满。

**这是 vLLM / SGLang 吞吐高的基础,也是本仓库 mini-vllm 已实现的核心。**

## 4. 优化手段二:投机解码(Speculative Decoding)—— 小模型猜,大模型验

**思路:** 用一个快的小模型(draft model)先猜 K 个 token,大模型(target)一次前向**并行验证**这 K 个 token:

1. draft 生成候选序列;
2. target 一次前向,同时校验这 K 个 token 是否与自己的分布一致;
3. 接受一致的 token,从第一个不一致处截断,重新从那里继续。

**为什么有效:** 大量 token 是"很容易预测"的(`的`、`了`、代码缩进、常见搭配)。小模型猜中率高;猜中的 token 只花了 1 次大模型前向,却"白赚"了 K 个 token。

**收益/代价:**

- decode 提速 2–3 倍(取决于接受率),**且输出分布与 target 自回归严格等价**(无损);
- 只改善 TPS / ITL,**不改善 TTFT**(prefill 没被加速);
- 只在**低 batch** 时有效——batch 大了,验证本身就把算力吃满,投机反而要动态关闭;
- 接受率受温度影响:温度越高分布越难猜,投机效果越差。

**RL 里的坑:** GRPO 需要 **old_logprobs**。投机解码不改输出分布,logprob 仍可算,但要保证 logprob 口径与 rollout 后端一致(本仓库 `MiniVllmRolloutWorker` 单独回算 old_logprobs,正是为对齐这种口径)。

## 5. 优化手段三:量化(Quantization)—— 省显存、提带宽

**思路:** 把权重/激活/KV cache 从 fp16/bf16 压到 int8/int4/fp8,用更低精度存和算。

**为什么对 decode 尤其有用:** decode 是带宽瓶颈,权重精度减半 → 每 token 从内存读的数据减半 → **等效内存带宽翻倍**;prefill 则是低精度 Tensor Core FLOPS 翻倍。

**三个量化层级(敏感性从低到高):**

| 层级 | 敏感性 | 说明 |
| --- | --- | --- |
| 权重(线性层) | 低 | 最安全,普遍量化;首尾层可保留原精度 |
| 激活 | 中 | 占内存小,很少单独量化 |
| **KV cache** | 中高 | RL 长上下文的大头;量化它省显存最多,但误差逐 token 累积 |
| attention(softmax 等) | 高 | 几乎不量化,保持原精度 |

**为什么 KV cache 量化风险高:** 每个新 token 的 attention 都依赖前面所有 token 的 KV,量化误差会**逐 token 累积**,序列越长误差越大。

**RL 特有坑:** GRPO 的 loss 对 logprob 精度敏感,量化误差会传导到 advantage。常见做法是 **bf16 训练 + 量化推理**,并单独验证量化前后 reward/logprob 一致性。

**常用格式:** int8(W8A8)、int4(AWQ/GPTQ)、fp8(E4M3);量化粒度(per-tensor → per-channel → per-block)越细越保质量,但 scale 开销越大。

## 6. 优化手段四:PD 分离(Disaggregated Serving)—— prefill 和 decode 拆到不同卡

**思路:** prefill 是 compute-bound、decode 是 memory-bound,需求不同。把它们拆到**不同的 GPU/节点**上,各自独立扩缩、互不抢占。

**一次请求怎么拆到两张卡(接力,不是并行算同一份工作):**

```
请求(prompt 几万 token)
   │
   ▼
① prefill 卡(compute-bound)
   └─ 整个 prompt 前向一次 → 产出 KV cache + 第一个 token
   └─ 把 KV cache 打包,通过高速互联(RDMA/InfiniBand)发给 decode 卡
   │        ↑ 这就是"接力棒"
   ▼
② decode 卡(memory-bound)
   └─ 收到 KV cache + 首 token,开始逐 token 生成
   └─ 后续每个 token 都用这张"从 prefill 卡传来的 KV cache"
```

关键点:

- **KV cache 是中间产物**:prefill 卡算完 prompt 后,把"prompt 里每个 token 的 K/V 向量"打包传过去。decode 卡不需要重新读 prompt,直接接着生成。
- **KV cache 很大**:一个长 prompt 的 KV cache 可能几百 MB 到几 GB,所以 PD 分离一般要求数据中心内网/InfiniBand,否则传输时间比省下来的还多。
- **请求只在 decode 卡上"逐 token 变慢"**:decode 卡接手的瞬间,TPS 就决定了整个请求后半段的体验。

**为什么值得:** 高流量下 prefill 和 decode 同卡会互相干扰——大 prompt 的 prefill 会打断 decode 的稳定节奏,造成尾延迟抖动。分离后:
- prefill 卡专心批量处理新请求,算力吃满,不被逐 token 的慢节奏拖累;
- decode 卡不被大 prompt 的 prefill 打断,尾延迟稳定;
- 每个引擎可以独立调优(如 prefill 用更低 TP,decode 用更高 TP)。

**条件式分离(实际部署的优化):** 不是所有请求都走接力——如果请求短、或 prompt 前缀在 decode 卡上已有缓存,decode 卡**本地直接 prefill+decode**,不绕一圈,省掉互联传输。

**注意区分:PD 分离 ≠ trainer/rollout 解耦。** 两者结构相似(都是接力+重叠),但传的"棒子"完全不同:

| | 传什么 | 谁传给谁 | 属于 |
| --- | --- | --- | --- |
| PD 分离 | **KV cache**(推理中间态) | prefill 卡 → decode 卡 | 推理阶段内部 |
| trainer/rollout 解耦 | **权重 state_dict**(训练产物) | trainer 卡 → rollout 卡 | 训练与推理之间 |

PD 分离拆的是**推理内部**两个前后相接的阶段(prefill 在前、decode 在后),两者都是推理;trainer/rollout 解耦拆的是**训练与推理**两个大阶段(trainer 卡做参数更新,rollout 卡做生成),通过流水线重叠避免互相等待。二者只是共享"把不同负载拆开、让 GPU 不互相等"的一般思想,不是同一回事——PD 分离始终在推理阶段内部。

## 7. 其他 RL 特有 / 常用的 rollout 优化

| 手段 | 一句话 | 本仓库状态 |
| --- | --- | --- |
| **前缀缓存(prefix caching)** | 相同 prompt 前缀共享 KV,跳过重复 prefill | 未实现(数学题同前缀场景收益大) |
| **异步流水线 + 一代滞后** | trainer 训 batch k 时,rollout 生成 batch k+1,最多 lag 一代 | ✅ 已实现(双 GPU prefetch 对照) |
| **复用生成做 old_logprob** | response 的 logprob 在训练前向时顺带算,避免二次前向 | ✅ 已实现(`MiniVllmRolloutWorker`) |
| **length bucketing / 分桶** | 相同长度请求分桶,减少 padding 浪费 | ✅ 已实现(performance 文档有对照) |
| **投机解码** | 小模型猜、大模型验,decode 提速且无损 | ❌ 未实现 |
| **KV cache 量化** | 长上下文省显存大头,注意误差累积 | ❌ 未实现 |
| **PD 分离部署** | prefill/decode 拆卡,吞吐↑ 尾延迟↓ | 部分(双 GPU 解耦,非生产分离) |

## 8. 面试话术:把这些串成一条逻辑线

> "RL 里 rollout 占一个 iteration 的 80–90%,而 rollout 慢的根本原因是 decode 阶段 memory-bound——每生成一个 token 都要把全部权重从 HBM 读一遍,算力闲置在等带宽。
>
> 所以优化 rollout 就是围绕'让 decode 更省、更快、更不互相干扰':连续批处理让 GPU 每步都在算;投机解码用小模型猜、大模型验,一次前向多赚几个 token(只对低 batch 有效);量化把权重和 KV cache 压到 int8/int4,decode 带宽需求直接减半(KV cache 量化要小心逐 token 误差累积);PD 分离把 compute-bound 的 prefill 和 memory-bound 的 decode 拆到不同卡,互不抢占。
>
> 我在自己的 mini-verl 里实现了连续批处理和 trainer/rollout 的异步流水线,并实测 rollout 占 90%、decode 内部又占 rollout 一半,这正好验证了上面这条优化路径——下一个自然的方向就是投机解码和 KV cache 量化。"

## 9. 参考来源

- Philip Kiely, *Inference Engineering*(Baseten 免费电子书);Pragmatic Engineer 深潜访谈 [What is inference engineering?](https://newsletter.pragmaticengineer.com/p/what-is-inference-engineering)
- 本仓库实测: [mini-verl 性能实验](../engineering/mini-verl-performance.md)(rollout 90.5% 占比、prefill/decode 分解、group size 对照、双 GPU pipeline)
- 算法侧: [RL 速成](../guides/rl-crash-course.md)
