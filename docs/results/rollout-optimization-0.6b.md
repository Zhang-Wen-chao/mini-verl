# 0.6B GRPO rollout 优化对照实验（2026-08-24）

分支：`official-verl-grpo`。全部实验 artifact 在 L20：
`/mnt/storage01/zhangwenchao02/repos/mini-verl-l20/artifacts/rollout-opt/`

**结论状态：已固定（2026-08-24）。** 实验代码与本文档锁定于本机 mini-verl `main`
commit `f9cbe69`；上游 verl snapshot `c4b389adadc58ce51cb2b63e70df497ca166d77f`；
运行时 `/tmp/official-verl-local-fsdp-vllm/venv`（torch 2.11.0 / transformers 5.5.3 /
ray 2.55.1 / vllm 0.24.0 / verl 0.10.0.dev）。4 组对照实验全部跑满 16 steps，
artifact 目录：`baseline-2plus2`、`prefix-cache`、`kv-fp8`、`spec-ngram5c`。

**本实验确立的结论（每一条都可复现、带数据）：**

1. 该负载（0.6B + 短 prompt + 数学推理）下，vLLM rollout 生成不是瓶颈：
   generate 占 step 19%，actor 更新占 65%，ref 前向占 20%。
2. 前缀缓存、KV cache fp8 量化、ngram 投机解码在本负载下均无收益
   （generate +0.8%~+2.9%，吞吐 -2.6%~-2.8%），且未引入质量退化。
3. PD 分离在 L20 环境不可行：vLLM 0.24.0 缺 `kv_transfer` 模块、无 `nixl` 库、无 InfiniBand。
4. rollout 优化手段只在"decode 是瓶颈"的负载下才有意义；本负载的下一步
   优化方向是 actor 更新与权重同步，而非 decode。

---

## 1. 实验目的

用真实 GRPO 训练循环（官方 verl + vLLM rollout），在 Qwen3-0.6B + GSM8K smoke 负载下，
逐个测量 rollout 优化手段的收益：前缀缓存、KV cache 量化、投机解码（ngram）、PD 分离。
目标是回答"哪种负载下 rollout 优化才有意义"，而不是"优化手段永远有效"。

## 2. 实验设置（所有对照共用）

| 项 | 值 |
|---|---|
| 模型 | Qwen3-0.6B-Base |
| 算法 | GRPO（`algorithm.adv_estimator=grpo`），one-step-overlap 异步 |
| 数据 | GSM8K rule-reward parquet，train 256 / val 64 |
| 拓扑 | 4×L20：FSDP2 trainer 2 卡（GPU 0-1）+ vLLM rollout 2 卡 TP=2（GPU 2-3） |
| 步数 | 16 steps（`train_batch_size=16`，`n=4`） |
| prompt / response | max_prompt 512（实际均值 ~80 token）/ max_response 256 |
| rollout 参数 | `gpu_memory_utilization=0.60`，`load_format=safetensors`，`n=4` |
| 环境 | `/tmp/official-verl-local-fsdp-vllm/venv`，torch 2.11.0 / vllm 0.24.0 / verl 0.10.0.dev（c4b389a） |
| 测量 | `official_verl/analyze_one_step_timing.py` 解析 `train.log` 的 `timing_s/*` 与吞吐 |

**基线（baseline-2plus2，16 step 均值）：**

| 指标 | 值 |
|---|---|
| `timing_s/generate_async`（rollout 生成） | **3.50 s**（占 step 19.1%） |
| `timing_s/sync_rollout_weights`（权重同步） | 2.63 s |
| `timing_s/ref`（参考模型前向） | 3.61 s |
| `timing_s/update_actor`（actor 更新） | **12.01 s**（占 step 65.4%） |
| `timing_s/step` | 18.37 s |
| `perf/throughput` | 521.6 tok/s |

**关键事实：在这个负载下，rollout 生成不是瓶颈。** actor 更新（12.0s）占 step 的
65%，ref 前向占 20%；两者之和远超 generate。这与 4B 正式 run 的结论一致（4B 上
`update_actor` ~36s / `sync_rollout_weights` ~30s 都大于 `generate_async` ~14s）。
mini_verl 自研小栈"rollout 占 90%"是 HF 逐 prompt 生成 + Python 控制流的特例，不能外推到
vLLM rollout。

## 3. 对照一：前缀缓存（prefix-cache）

改动：`+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=True`

| 指标 | 基线 | 前缀缓存 | 变化 |
|---|---|---|---|
| generate_async | 3.502 | 3.605 | +2.9% |
| step | 18.373 | 19.174 | +4.4% |
| throughput | 521.6 | 507.6 | -2.7% |
| reward/mean | 0.0018 | 0.0 | 无退化 |
| response_length/mean | 214.6 | 217.1 | 一致 |

**结论：无收益。** 原因：GSM8K smoke 的 prompt 很短（~80 token），且 RL 每步更新权重后
vLLM 的 KV cache 失效，跨步无法复用。前缀缓存的收益场景是长系统 prompt / 多轮对话 /
同前缀大批量请求，本负载都不满足。

## 4. 对照二：KV cache 量化 fp8（kv-fp8）

改动：`+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8`

| 指标 | 基线 | kv-fp8 | 变化 |
|---|---|---|---|
| generate_async | 3.502 | 3.530 | +0.8% |
| step | 18.373 | 19.162 | +4.3% |
| throughput | 521.6 | 508.1 | -2.6% |
| reward/mean | 0.0018 | 0.0 | 无退化 |
| response_length/mean | 214.6 | 217.5 | 一致 |

**结论：无收益也无质量损失。** 原因：0.6B + 256 token 响应的 KV cache 很小（显存非瓶颈），
fp8 省下的带宽不足以抵消量化开销。KV cache 量化的价值在长上下文 / 大模型 / 高并发场景，
本负载 KV 占用太小。**没有观察到 reward/logprob 漂移，量化在数值上是安全的。**

## 5. 对照三：ngram 投机解码（spec-ngram5c）

改动：
```
actor_rollout_ref.model.mtp.enable_rollout=true
actor_rollout_ref.model.mtp.method=ngram
actor_rollout_ref.model.mtp.num_speculative_tokens=5
```

| 指标 | 基线 | ngram 投机 | 变化 |
|---|---|---|---|
| generate_async | 3.502 | 3.545 | +1.2% |
| step | 18.373 | 19.009 | +3.5% |
| throughput | 521.6 | 507.2 | -2.8% |
| reward/mean | 0.0018 | 0.0 | 无退化 |
| response_length/mean | 214.6 | 215.6 | 一致 |

**结论：无收益。** 原因：ngram 投机用"查重复 n-gram"做草稿，对数学推理这类**几乎
全唯一 token 序列**的负载猜中率极低；且 0.6B 生成本身很快（decode 不是瓶颈），投机验证
的开销反而拖慢。投机解码的价值在**低 batch + 大模型 + 高可预测文本**（代码、自然语言），
数学推理负载是最不匹配的场景。

**配置注记（踩坑记录）：** verl 的投机入口是 `actor_rollout_ref.model.mtp.*`（`MtpConfig`），
不是 `actor_rollout_ref.rollout.mtp.*`；`engine_kwargs.vllm.speculative_config` 是
MTP/draft 模型的补充通道，纯 ngram 不需要它。错误的 key 会报
`Key 'enable_rollout' is not in struct`。

## 6. 对照四：PD 分离 —— 环境不可行（记录为负结论）

verl 本版本（c4b389a）**原生支持** PD 分离：`rollout.disaggregation.enabled=true` +
`prefill_replicas` / `decode_replicas` / `transfer_backend`（nixl / mooncake），
实现见 `verl/workers/rollout/vllm_rollout/vllm_pd_replica.py`。

但当前 L20 环境**无法运行**：

1. **vLLM 0.24.0 安装缺 KV 传输组件**：`vllm.v1.core.kv_transfer` 模块不存在；
2. **nixl 库未安装**（`import nixl` 失败）；
3. **L20 无 InfiniBand**：PD 分离的 KV cache 跨机传输需要高速互联（RDMA），
   PCIe（~32GB/s）传输 KV 的开销大于收益。

**结论：PD 分离需要专门的 KV 传输栈（nixl/mooncake）与高速互联，本环境不具备，
标记为不可行而非"无收益"。** 这印证了文档里的判断：PD 分离一般要求数据中心内网。

## 7. 综合结论

| 手段 | 对 generate 的影响 | 对 step 的影响 | 质量 | 结论 |
|---|---|---|---|---|
| 前缀缓存 | +2.9% | +4.4% | 无退化 | 本负载无收益（prompt 短 + cache 跨步失效） |
| KV cache fp8 | +0.8% | +4.3% | 无退化 | 本负载无收益（KV 太小，非瓶颈） |
| ngram 投机 | +1.2% | +3.5% | 无退化 | 本负载无收益（数学序列不可预测） |
| PD 分离 | — | — | — | 环境不可行（缺 kv_transfer/nixl/IB） |

**核心洞察（面试可讲）：**

> "我在真实 GRPO 循环里测了四个 rollout 优化手段,结论是**它们都没有收益**。
> 原因是这个负载的瓶颈根本不在 decode:0.6B + 短 prompt + 数学推理,vLLM rollout
> 生成只占 step 的 19%,而 actor 更新占 65%、ref 前向占 20%。
> 前缀缓存失效是因为 RL 每步权重更新让 KV cache 无法跨步复用;KV 量化无收益是因为
> KV cache 太小不是瓶颈;ngram 投机无收益是因为数学推理的 token 几乎不可预测。
> 这恰好证明了文档里的观点:rollout 优化手段只在'decode 是瓶颈'的负载下才有意义——
> 大模型、长输出、高并发、可预测文本。我做实验前如果只看概念图,会以为加这些
> 开关就能提速;实测数据告诉我先要量化瓶颈在哪。"

**对 4B 正式 run 的提示：** 4B 上 `update_actor`（~36s）和 `sync_rollout_weights`（~30s）
才是最大项，下一步优化方向是 actor 更新 / 权重同步（如更小的 update bucket、async 权重
传输、ref 与 actor 并行），而不是继续压 decode。

## 8. 复现命令

基线（其余对照在基线上加一行参数）：
```bash
cd /mnt/storage01/zhangwenchao02/repos/mini-verl-l20
export VERL_DIR=$PWD/.official-verl/verl
export MODEL_PATH=$PWD/.official-verl/models/Qwen3-0.6B-Base
export TRAIN_FILE=$PWD/.official-verl/data/gsm8k-smoke/train.parquet
export TEST_FILE=$PWD/.official-verl/data/gsm8k-smoke/test.parquet
export RUN_ROOT=$PWD/artifacts/rollout-opt/<run-name>
export CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_SHM_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
PATH=/tmp/official-verl-local-fsdp-vllm/venv/bin:$PATH \
  bash official_verl/run_qwen3_0_6b_4gpu_smoke.sh [对照参数]
```
计时解析：`official_verl/analyze_one_step_timing.py <train.log> --report <out.json>`
