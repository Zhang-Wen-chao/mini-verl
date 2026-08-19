# mini-verl 性能记录

## 2026-08-19：官方 verl 4-GPU GRPO smoke（完成）

L20 上使用锁定为 `c4b389adadc58ce51cb2b63e70df497ca166d77f` 的官方 `verl`
codeload 快照、Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`
与 GSM8K `256/64` train/test parquet，完成了一次真实官方 GRPO 训练。拓扑是
GPU 0/1 的 FSDP2 actor trainer 加 GPU 2/3 的 TP=2 vLLM rollout；总共 16 个
optimizer steps。`global_step_16` 的两份模型分片各 1,503,446,571 bytes、两份
optimizer 分片各 2,384,226,344 bytes。训练后 driver 在清理阶段自然退出，四卡
均恢复到 `4 MiB / 0%`。

运行时是可从同一 pinned source + `uv.lock` 重建的容器本地环境
`/tmp/official-verl-local-fsdp-vllm`，而模型、数据和产物在持久盘。该选择仅为
规避共享文件系统上 CUDA 包解压极慢；它没有改变源码或 lock。锁定 preflight
确认 Python 3.12、Torch `2.11.0+cu130`、Transformers `5.5.3`、Ray `2.55.1`、
vLLM `0.24.0` 以及四张 46,068 MiB L20 均可用。

末步实测：训练 reward 均值 `0.015625`、actor KL `0.0007783`、actor loss
`0.0002129`、吞吐约 `914.8 tokens/s`、held-out GSM8K accuracy `0`。回答平均
255.75/256 tokens，clip ratio `0.96875`，因此这次只验收为基础设施 smoke，
不作为模型质量或 GRPO 收敛结论。4B 之前必须先使用非截断响应长度、测量显存
余量与非退化 reward 分布，再做 held-out 对照。完整运行记录位于持久化 artifact
目录的 `RUNLOG.md`。

## 2026-08-18：toy GRPO Controller

### 目的与边界

本记录用于验证 `mini-verl` 的同步 `Controller` 闭环、指标采集与 CUDA allocator 峰值显存观测可复现。工作负载是极小的 categorical toy policy，不含真实 Transformer prefill/decode、KV cache、通信或服务调度，**不能**解释为 LLM 训练/推理吞吐。

### 环境

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA L20（使用 `CUDA_VISIBLE_DEVICES=0`） |
| NVIDIA driver | 550.127.05 |
| PyTorch | `2.10.0a0+a36e1d39eb.nv26.01.42222806` |
| Transformers | `5.9.0` |
| 测试日期 | 2026-08-18 |

运行前后均通过 `nvidia-smi` 确认四张 L20 无计算进程；完成后每卡为 4 MiB / 0% utilization。

### 命令

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python benchmarks/toy_grpo_benchmark.py \
  --device cuda --iterations 100 --warmup 1 --repeats 3
```

输出为单条 JSON，便于后续存档或对比：

```json
{
  "benchmark": "toy_grpo_controller",
  "cuda_max_allocated_bytes_median": 21504,
  "cuda_max_reserved_bytes_median": 2097152,
  "device": "cuda",
  "gpu_memory_used_max_mib_max": 341,
  "gpu_physical_index": 0,
  "gpu_utilization_interval_seconds": 0.1,
  "gpu_utilization_max_percent_max": 9,
  "gpu_utilization_mean_percent_median": 8.889,
  "gpu_utilization_sample_count_median": 9,
  "gpu_utilization_scope": "device_level_nvidia_smi",
  "iteration_seconds_median": 0.008874,
  "iterations": 100,
  "iterations_per_second": 111.688,
  "repeats": 3,
  "run_seconds_median": 0.895349,
  "warmup": 1
}
```

### 解释

- `iteration_seconds_median` 是每次重复中最后一次 Controller iteration 的中位值；它用于粗粒度回归检测，不是 p50 的完整迭代分布。
- `run_seconds_median` 与 `iterations_per_second` 是 100 次 iteration 整体运行时间的三次重复中位数。
- `cuda_max_allocated_bytes_median` / `cuda_max_reserved_bytes_median` 是 PyTorch caching allocator 的测量区间峰值，不等于模型显存，也不等于 `nvidia-smi` 的进程显存。
- GPU utilization 以 0.1 s `nvidia-smi` 设备级采样得到。运行前已确认 GPU 0 无其他计算进程，因此该次采样可归因给此工作负载；它不是进程级利用率，不能直接迁移到共享卡或其他模型。

## 2026-08-18：tiny GPT-2 Hugging Face GRPO Controller

### 目的与配置

该基准不下载权重，而是在本地构造 `GPT2LMHeadModel`（vocab=64、hidden=64、2 layers、2 heads），真实执行 Hugging Face `generate`、旧策略 response logprob 回算、轨迹封装、rule reward / GRPO advantage 与 CausalLM GRPO 反传。它验证的是框架真实 backend 的阶段归因，仍然不是业务模型或服务吞吐测试。

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python benchmarks/tiny_hf_grpo_benchmark.py --device cuda \
  --iterations 20 --warmup 1 --repeats 3 \
  --prompt-count 4 --group-size 2 --max-new-tokens 8 \
  --sample-gpu-utilization --gpu-index 0
```

运行前确认 GPU 0 无计算进程；阶段计时在每个阶段前后调用 `torch.cuda.synchronize()`，因此包含本阶段已提交的 CUDA 工作，而不是异步 launch 时间。

```json
{
  "benchmark": "tiny_hf_grpo_controller",
  "cuda_max_allocated_bytes_median": 21307904,
  "cuda_max_reserved_bytes_median": 27262976,
  "decode_forward_calls_median": 28.0,
  "decode_seconds_median": 0.028703,
  "group_size": 2,
  "iteration_seconds_median": 0.073326,
  "iterations_per_second": 13.638,
  "max_new_tokens": 8,
  "prefill_forward_calls_median": 4.0,
  "prefill_seconds_median": 0.004492,
  "prompt_count": 4,
  "response_tokens_median": 63.0,
  "response_tokens_per_second": 859.177,
  "reward_seconds_median": 0.000216,
  "rollout_seconds_median": 0.066361,
  "train_seconds_median": 0.006783
}
```

### 归因

- rollout 占同步 iteration 中位数约 **90.5%**：它包括 token-by-token `generate` 和为了记录 `old_logprobs` 的额外 CausalLM forward；
- train 占约 **9.3%**；reward 占约 **0.3%**；其余是 Controller 记账开销；
- generate 内部针对 4 个 prompt 记录到 **4 次 prefill**（4.492 ms）和 **28 次 decode**（28.703 ms）：每个 prompt 一次 prefill，随后为生成 8 个 token 进行 7 次 cache decode。两者合计 33.195 ms，约为 rollout 的一半；其余 rollout 时间来自 HF generation 控制流、采样/轨迹构造和旧策略 logprob 回算。
- allocator 峰值约 20.3 MiB allocated / 26.0 MiB reserved，`nvidia-smi` 设备级显存峰值为 385 MiB；
- 在确认独占的 GPU 0 上，0.1 s 设备级采样平均 utilization 为 12.929%、峰值 13%、中位 14 个样本。

这恰好揭示下一步优化方向：不能只调 GRPO loss；对真实规模模型需要优先替换逐 prompt 的 HF `generate` / 额外 logprob forward，研究 continuous batching、vLLM/SGLang rollout、prefill/decode 分离和 trainer-rollout 资源解耦。

### group size 对照：2 → 4

只把 `group_size` 从 2 改为 4，其余模型、4 个 prompts、`max_new_tokens=8`、20 iterations、1 次 warmup、3 次重复、CUDA 同步阶段计时和 GPU 0 设备级采样均不变。运行前同样确认 GPU 0 无其他计算进程。

| 指标 | group=2 | group=4 | 变化 |
| --- | ---: | ---: | ---: |
| response tokens / iteration | 63.0 | 121.5 | +92.9% |
| iteration median | 73.326 ms | 77.093 ms | +5.1% |
| response token throughput | 859.177 tok/s | 1576.019 tok/s | +83.4% |
| rollout median | 66.361 ms | 67.696 ms | +2.0% |
| train median | 6.783 ms | 8.973 ms | +32.3% |
| generate prefill | 4.492 ms / 4 calls | 4.471 ms / 4 calls | 基本持平 |
| generate decode | 28.703 ms / 28 calls | 28.573 ms / 28 calls | 基本持平 |
| allocator allocated peak | 20.3 MiB | 21.7 MiB | +7.0% |
| device utilization mean | 12.929% | 12.867% | 基本持平 |

`group=4` 的完整结果：

```json
{
  "benchmark": "tiny_hf_grpo_controller",
  "cuda_max_allocated_bytes_median": 22800896,
  "cuda_max_reserved_bytes_median": 27262976,
  "decode_forward_calls_median": 28.0,
  "decode_seconds_median": 0.028573,
  "group_size": 4,
  "iteration_seconds_median": 0.077093,
  "iterations_per_second": 12.971,
  "prefill_forward_calls_median": 4.0,
  "prefill_seconds_median": 0.004471,
  "response_tokens_median": 121.5,
  "response_tokens_per_second": 1576.019,
  "reward_seconds_median": 0.000382,
  "rollout_seconds_median": 0.067696,
  "train_seconds_median": 0.008973
}
```

在这个很小的合成模型上，group 增大接近翻倍的有效 token 数，却没有使 rollout 或 generate 内部 prefill/decode 时间等比例上升：每个 prompt 的 generate 调用次数不变，只是 batch 从 2 扩为 4，固定的调度和 Python 开销被摊薄。它是一个 **fixed-shape regression observation**，不是“group size 总应取更大”的结论：真实模型还要权衡 reward 方差、显存、上下文长度、并行度与服务批处理能力。

## 2026-08-19：双 GPU trainer/rollout 同步与 one-step-lag 对照

### 设置

使用两个彼此独立的本地构造 tiny GPT-2 副本（vocab=64、hidden=64、2 layers、2 heads）：trainer 固定在 L20 GPU 0，rollout 固定在 L20 GPU 1。每轮均以全量跨设备 `state_dict` 拷贝发布训练后的策略。同步模式是 `rollout → reward → train → sync`；prefetch 模式在训练当前 batch 时，以独立 rollout 副本生成同版本后继 batch，且最多允许一代 policy lag。

两组均为 4 prompts、`group_size=2`、`max_new_tokens=8`、20 个**稳态** iteration、1 次 warmup、3 次重复；prefetch 的 prime rollout 不计入每轮 wall time。运行前已确认 GPU 0 和 GPU 1 没有其他计算进程。

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 \
python benchmarks/tiny_hf_pipeline_benchmark.py --pipeline synchronous \
  --trainer-device cuda:0 --rollout-device cuda:1 \
  --iterations 20 --warmup 1 --repeats 3 --prompt-count 4 \
  --group-size 2 --max-new-tokens 8 --sample-gpu-utilization \
  --trainer-gpu-index 0 --rollout-gpu-index 1

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 \
python benchmarks/tiny_hf_pipeline_benchmark.py --pipeline prefetch \
  --trainer-device cuda:0 --rollout-device cuda:1 \
  --iterations 20 --warmup 1 --repeats 3 --prompt-count 4 \
  --group-size 2 --max-new-tokens 8 --sample-gpu-utilization \
  --trainer-gpu-index 0 --rollout-gpu-index 1
```

### 结果与归因

| 指标 | 同步 | 一代 lag prefetch | 变化 |
| --- | ---: | ---: | ---: |
| 稳态 iteration wall time | 72.928 ms | 69.000 ms | -5.4% |
| response token throughput | 829.585 tok/s | 876.812 tok/s | +5.7% |
| rollout 总 wall time | 64.730 ms | 67.584 ms | 近似持平 |
| train + reward | 7.028 ms | 11.415 ms | 可重叠窗口 |
| full state-dict sync | 1.234 ms | 1.274 ms | 近似持平 |
| prefetch 有效重叠 | 0 ms | 11.417 ms | rollout 的约 16.9% |
| prefetch 未隐藏 rollout 尾部 | 64.730 ms | 56.125 ms | 仍是主关键路径 |
| trainer / rollout allocator peak | 20.3 / 10.1 MiB | 20.3 / 10.1 MiB | 持平 |

prefetch 的完整 JSON（重测时增加了 `next_rollout_wall_seconds` 字段）：

```json
{
  "benchmark": "tiny_hf_grpo_two_gpu_pipeline",
  "pipeline": "prefetch",
  "iteration_wall_seconds_median": 0.069,
  "next_rollout_wall_seconds_median": 0.067584,
  "prefetch_overlap_seconds_median": 0.011417,
  "rollout_wait_seconds_median": 0.056125,
  "sync_seconds_median": 0.001274,
  "response_tokens_per_second": 876.812,
  "trainer_gpu_utilization_mean_percent_median": 1.5,
  "rollout_gpu_utilization_mean_percent_median": 12.714
}
```

这是一个有价值的**负/边界结果**：调度与版本语义都正确，双卡确实产生了可度量的 11.4 ms overlap，但 tiny workload 的 train 阶段远短于 rollout，最多只能隐藏约 16.9% rollout。因此只把训练和推理解耦到不同 GPU 并不能解决主瓶颈。对真实模型，若训练时间更长、group/batch 更大，或 rollout 改用 continuous batching 引擎，重叠潜力会增大；若仍是生成主导，则优先优化 decode、old-logprob 回算和 serving backend。

## 2026-08-19：rollout 细分补充与真实 CausalLM length bucketing

### 更新后的 rollout 细分

同一 tiny GPT-2 HF GRPO 配置（4 prompts、`group_size=2`、`max_new_tokens=8`、20 iterations、1 次 warmup、3 次重复）新增了 sampled-policy `old_logprob` 的整批 CausalLM forward 计时；只在诊断 benchmark 中启用 CUDA 同步。最新结果如下：

| rollout 子阶段 | 中位耗时 | 调用数 / 说明 |
| --- | ---: | --- |
| generate prefill | 4.485 ms | 4 calls；每个 prompt 一次 |
| generate decode | 28.440 ms | 28 calls；每个 prompt 生成 8 token 后有 7 次 cache decode |
| old-logprob forward | 1.430 ms | 对全部 sampled trajectories 的单次完整 CausalLM forward |
| rollout total | 65.978 ms | 含 HF generation 控制流、采样、输入/trajectory 打包 |

因此，已直接测得的 model forward 时间为 34.355 ms；剩余约 31.623 ms（47.9% rollout）来自框架侧生成控制流和 Python/数据处理。这个 tiny 模型不能给出业务绝对吞吐，但它清楚指出：替换 correctness-first 的逐 prompt HF generate、降低 Python 调度与做 serving continuous batching，是比优化 GRPO loss 更优先的方向。

### 受控 length-bucketing 对照

两种策略使用完全相同的 32 条 trajectory：prompt 恒为 4 token；response lengths 为 2、4、8、16，各 8 条。两边 batch size=4、20 epochs、每 epoch 8 steps，因此均为 **160 optimizer steps、7,360 real sequence tokens、4,800 response tokens**。唯一变化是 batch 内组成：`mixed` 混合四种长度，`bucketed` 按长度分桶。模型为本地构造 tiny GPT-2；GPU 0 运行前确认无其他计算进程；每种策略 1 次 warmup、3 次重复。

| 指标 | mixed | bucketed | 变化 |
| --- | ---: | ---: | ---: |
| padded sequence tokens | 12,800 | 7,360 | -42.5% |
| padding ratio | 42.5% | 0.0% | -42.5 pp |
| train wall time | 929.495 ms | 877.213 ms | -5.6% |
| real sequence token throughput | 7,918.278 tok/s | 8,390.209 tok/s | +6.0% |
| response token throughput | 5,164.094 tok/s | 5,471.875 tok/s | +6.0% |
| allocator allocated peak | 20.19 MiB | 20.18 MiB | 持平 |
| device utilization mean | 11.600% | 12.556% | 近似持平 |

padding 降幅没有 1:1 转化成时延收益，原因是 optimizer、kernel launch 和 Python 调度仍有固定成本；但在真实 token 数与 step 数严格不变时，6.0% 的 real-token 吞吐提升说明 length bucketing 是有效的工程优化。评估变长训练时应始终同时报告 padding ratio、real-token throughput 和 optimizer step 数，而不是只看 padded FLOPs 或 batch/s。

## 2026-08-19：Hugging Face rollout prompt micro-batching

### 实现

`HuggingFaceRolloutWorker` 新增显式的 `rollout_batch_size`（默认 `1` 保持原语义）。大于 1 时，worker 逐 prompt 编码、在 worker 内**左填充**后一次调用 `model.generate`；每个返回 trajectory 仍保存未填充的原 prompt ids、原 metadata、正确的 group id 和与 response 一一对应的 old logprob。它是 correctness-first 的静态 prompt micro-batch，不是 continuous batching。

真实 CUDA 回归同时覆盖了：

- 多 prompt × `num_return_sequences` 的返回顺序与 group/metadata 归属；
- 长短不同 prompt 的左填充，确保 padding 不泄漏进 trajectory prompt ids；
- 重新计算的 old logprob 与每条 response token 的长度对齐。

### 受控性能对照

固定 tiny GPT-2、4 prompts、`group_size=2`、`max_new_tokens=8`、20 iterations、1 次 warmup、3 次重复和相同 CUDA 同步诊断口径，仅改变 `rollout_batch_size=1 → 4`。GPU 0 在每组前确认无其他计算进程。

| 指标 | batch=1 | batch=4 | 变化 |
| --- | ---: | ---: | ---: |
| generate prefill calls / iteration | 4 | 1 | -75% |
| generate decode calls / iteration | 28 | 7 | -75% |
| generate prefill | 4.519 ms | 1.151 ms | -74.5% |
| generate decode | 28.848 ms | 7.245 ms | -74.9% |
| old-logprob forward | 1.437 ms | 1.436 ms | 持平 |
| rollout total | 66.807 ms | 19.487 ms | -70.8% |
| iteration total | 73.876 ms | 26.539 ms | -64.1% / **2.78×** |
| response token throughput | 852.780 tok/s | 2,355.025 tok/s | **+176.2%** |
| allocator allocated peak | 20.32 MiB | 20.32 MiB | 持平 |

这不是靠降低采样工作量得到的：两边都生成约 63 response tokens/iteration。收益主要来自将 4 次 prompt-level generate 调用合并成 1 次 batch=8（`4 prompts × 2 samples`）生成，减少 Python/Transformers generation 控制流和重复 kernel launch。`old_logprob` 回算已是一个完整轨迹 batch 的 forward，故基本不变。

适用边界：静态 micro-batch 会受最长 prompt/response 的 padding、显存和请求到达时间限制；在线服务的更进一步收益需要 continuous batching 和 paged KV cache（vLLM/SGLang），这不在本项目当前的无新增依赖范围内。

### micro-batching 后的双 GPU prefetch 取舍

将同一 prompt micro-batch（`rollout_batch_size=4`）接入 GPU 0 trainer / GPU 1 rollout 的 pipeline benchmark 后，仍固定 4 prompts、`group_size=2`、`max_new_tokens=8`、20 个稳态 iteration、1 次 warmup、3 次重复：

| 指标 | 双 GPU 同步 | 双 GPU one-step-lag prefetch | 变化 |
| --- | ---: | ---: | ---: |
| iteration wall time | 27.444 ms | 23.096 ms | -15.8% |
| iteration throughput | 36.438 iter/s | 43.298 iter/s | +18.8% |
| response token throughput | 2,222.708 tok/s | 2,684.448 tok/s | +20.8%* |
| full rollout wall time | 19.096 ms | 21.779 ms | 近似同量级 |
| prefetch overlap | 0 ms | 11.358 ms | rollout 的约 52.1% |
| unhidden rollout tail | 19.096 ms | 10.411 ms | 仍在关键路径 |
| full cross-device state-dict sync | 1.235 ms | 1.243 ms | 持平 |

`*` 采样 response 的 EOS 长度有轻微波动（同步中位 61、prefetch 中位 62 tokens），因此主要判断应以同一任务下的 iteration wall time 为准。

这组组合实验给出一个更细的工程结论：先把逐 prompt 生成合并为 micro-batch，单卡最主要的框架开销已被去除；然后在独立 rollout GPU 上做安全的一代 lag 预取，仍可隐藏约一半 rollout、再获得约 15.8% 稳态 iteration 收益。是否值得第二张 GPU取决于资源成本和模型规模：在这里同步双卡相对单卡 batch=4 基线没有明显优势，而 prefetch 才把 wall time 压到 23.096 ms；真实大模型需以实际 rollout/train 比和权重同步成本重新判断。

### 正确性关联验证

同一 CUDA 环境下：

- `python -m unittest discover -s tests -v`：59/59 通过（含真实 HF rollout 的可选 prefill/decode/old-logprob hook、prompt micro-batching 左填充、双 CausalLM 一代 lag 与真实 CausalLM length-bucketing 对照）；
- toy GRPO end-to-end：greedy pass@1 从 `0.125` 到 `1.000`；
- 两卡 tiny GPT-2 DDP GRPO smoke：两个 rank 参数同步，mean loss `-0.02601364`。

下一份可用于性能结论的真实 LLM 报告，应固定模型完整快照、prompt/response 长度分布、group size、batch size、精度、rollout backend、并行策略、GPU 数量和预热/重复口径，且分别给出 prefill、decode、reward、train 和 policy-sync 的时间。

## 2026-08-19：rollout 侧 prompt-length bucketing（负结果也保留）

### 实现与测量口径

`HuggingFaceRolloutWorker` 新增可选 `bucket_prompts_by_length`。worker 对每个 prompt 只编码一次；开启时按 token length 降序组成既定 `rollout_batch_size` 的静态 micro-batch，再将生成的 trajectory 按原始 prompt index 与 sample index 排回稳定顺序。因此 `group_id`、原始未填充 `prompt_token_ids`、metadata 和 old-policy logprob 的下游契约均不变。

本次只测 rollout，不混入 reward 或 trainer。模型为本地构造 tiny GPT-2（vocab=64、hidden=64、2 layers、2 heads）；8 个 prompt 的原输入顺序故意交错为 `3,24,4,23,5,22,6,21` token；`group_size=2`、`max_new_tokens=8`、`rollout_batch_size=4`、20 iteration、1 warmup、3 repeats。两组仅改变是否开启分桶。GPU 0 在每组前确认无计算进程；计时以 CUDA 同步包住完整 rollout，generate 内 forward hook 仍给出 prefill/decode 分解。

| 指标 | 原顺序 micro-batch | length bucketed | 变化 |
| --- | ---: | ---: | ---: |
| real prompt tokens（含 group expansion） | 216 | 216 | 相同 |
| padded prompt tokens | 368 | 240 | -34.8% |
| prompt padding ratio | 41.3% | 10.0% | -31.3 pp |
| prompt batches / prefill calls | 2 / 2 | 2 / 2 | 相同 |
| prefill | 2.760 ms | 2.738 ms | -0.8% |
| decode | 18.079 ms | 17.974 ms | -0.6% |
| old-logprob forward | 1.388 ms | 1.381 ms | -0.5% |
| rollout wall time | 40.775 ms | 40.740 ms | -0.1% |
| response token throughput | 3,065.604 tok/s | 3,055.965 tok/s | -0.3%* |
| allocator allocated peak | 12.70 MiB | 12.69 MiB | 持平 |

`*` 两组采样产生的 response token 中位数为 125.0 / 124.5，EOS 随机波动足以解释该极小差异；应以相同 generation workload 下的 wall time 判断。设备级利用率均约 13%，峰值 14%，显存峰值均 375 MiB。

这个结果**不应**被表述为“bucketing 无效”。它证明实现确实消除了 128 个 generate-prefill prompt token，同时也证明该 tiny workload 的完整 rollout 主要由 decode 与 HF/Python 固定控制流主导，少量 prefill padding 不在关键路径。对长上下文、大 hidden size 或更大静态 batch，应以真实 prompt-length 分布重测；未来可将固定 prompt count 策略提升为 token-budget batching，使 bucket 的长度分布直接决定 batch 容量。

### 扩展 prompt-token budget：容量上限与吞吐取舍

在同一 worker 上新增 `rollout_max_padded_prompt_tokens`：对每个 `generate` 调用约束
`group_size × rows × max_prompt_length`，也就是 `num_return_sequences` 展开后 prefill 输入 tensor 的 token 数。它不估算 generated token 的 KV cache、模型 activation 或 decode 资源，因此是一个保守且可观察的 **prefill 准入上限**，不是完整的显存预测器。单条 prompt 展开后已经超过预算会在调用模型前报错。

为量化代价，固定相同 8 条交错长度 prompt、同一 tiny GPT-2、`group_size=2`、`max_new_tokens=8`、20 iteration、1 warmup、3 repeats 与 length bucketing；唯一改变是 count cap=8 下是否施加 180 expanded-prompt-token 上限。GPU 0 在每组前确认无计算进程。

| 指标 | 无 token budget | budget=180 | 含义 |
| --- | ---: | ---: | --- |
| prompt batches / prefill calls | 1 / 1 | 3 / 3 | 容量保护的拆批代价 |
| 单 batch padded prompt token 峰值 | 384 | 168 | -56.3%，并严格低于 180 |
| 累计 padded prompt tokens | 384 | 318 | -17.2% |
| prompt padding ratio | 43.75% | 32.08% | -11.67 pp |
| generate prefill | 1.352 ms | 3.915 ms | 额外 2 次调用 |
| generate decode | 9.040 ms | 25.388 ms | 额外 2 次调用 |
| rollout wall time | 22.924 ms | 56.847 ms | +148.0% |
| response token throughput | 5,365.556 tok/s | 2,137.316 tok/s | -60.2%* |
| allocator allocated peak | 12.70 MiB | 12.69 MiB | tiny 模型下持平 |

`*` 采样 response token 中位数为 123.0 / 121.5，有少量 EOS 随机波动；容量/调用数与 wall time 才是该对照的主结论。

结论是明确的工程取舍：budget 不是小模型上的吞吐优化；它牺牲 batch 合并度，把单次 prefill 容量从 384 限制到 168。真实大模型在长 prompt、较大 group size 或需要避免 OOM/尾延迟时，这个硬上限是必要控制面；应按照实际可用显存和 context window 配置，并结合 KV-cache/生成长度的进一步 admission 模型，而不是照搬此 180 的数字。

### 最坏 sequence-token budget：将 `max_new_tokens` 纳入 admission

仅限制 prefill token 仍会忽略 decode 期间的 KV-cache 增长。因此 worker 还提供 `rollout_max_padded_sequence_tokens`，其调度口径为：

```text
group_size × prompt rows × (batch max prompt length + max_new_tokens)
```

这里的 `max_new_tokens` 是为静态 rollout 预留的**最坏上界**，不是将 sampled response 的实际长度伪装成固定值。该 token 数也不是字节级显存模型：layer 数、hidden size、KV dtype、attention 实现、activation 和 allocator 行为均未被表达；它的作用是一个可解释的序列容量准入 guard。若单条 prompt 加预留生成长度也放不下，worker 在 `model.generate` 前拒绝请求。

沿用上节完全相同的 tiny GPT-2、8 条交错长度 prompt、`group_size=2`、`max_new_tokens=8`、count cap=8、length bucketing、20 iteration、1 warmup、3 repeats，仅配置 sequence budget=240（不配置 prompt budget）：

| 指标 | 无 budget | sequence budget=240 | 变化 / 含义 |
| --- | ---: | ---: | --- |
| prompt batches / prefill calls | 1 / 1 | 3 / 3 | 受 decode reservation 触发拆批 |
| 单 batch worst-case sequence token 峰值 | 512 | 232 | -54.7%，严格低于 240 |
| 单 batch prompt token 峰值 | 384 | 168 | 调度拆分的派生效果 |
| 累计 padded prompt tokens | 384 | 318 | -17.2% |
| rollout wall time | 22.924 ms | 56.331 ms | +145.7% |
| response token throughput | 5,365.556 tok/s | 2,156.894 tok/s | -59.8%* |
| allocator allocated peak | 12.70 MiB | 12.69 MiB | tiny 模型下持平 |

`*` response token 中位数为 123.0 / 121.5；EOS 随机波动存在，但本组主结论是 capacity 和调用次数，不是 token/s 的微小差异。

这进一步强调了容量控制与性能优化的分工：在 tiny GPT-2 上，三次 HF generate 的固定开销主导，故保守 admission 会明显降低吞吐；在真实长上下文或显存临界环境中，避免单 batch 的 sequence/KV 峰值越界通常比局部吞吐更重要。生产式 scheduler 还需把实际 KV-cache bytes、请求到达、早停、paged attention 和并发副本纳入模型；本项目的实现刻意保持为可验证的静态上界。

### Context window 的 fail-fast guard

batch token budget 不等同于模型可接受的单 request context。worker 现在在 `generate` 前检查每个 encoded prompt 的 `prompt length + max_new_tokens`；若模型 config 暴露 `max_position_embeddings`（或 GPT-2 的 `n_positions`），超过该上限的固定长度 request 会立即抛出包含所需 position、prompt 长度、`max_new_tokens` 与模型容量的 `ValueError`。L20 CUDA 测试覆盖了 4-token prompt + 5 new tokens 被 `n_positions=8` 明确拒绝的路径。

这个 guard 的边界同样重要：没有公开静态 context 字段的模型由其 backend 决定；动态 RoPE scaling 或服务端另外配置的可扩展 context 不应被这个最小实现擅自推断。生产环境应把模型部署配置作为 admission policy 的权威来源。

## 2026-08-19：sequence admission 在端到端 GRPO 闭环中的代价

前述容量实验只测 rollout。为避免把 rollout 局部时延误当成训练系统结论，`tiny_hf_grpo_benchmark.py` 现在也支持变长 synthetic prompt、length bucketing、prompt/sequence budget，并在每个完整 iteration 输出 `prompt_batch_count`、累计 padding 与单 batch prompt/sequence 峰值。它仍真实执行：HF `generate`、old-policy logprob CausalLM forward、rule reward/advantage 和 CausalLM GRPO update。

固定本地 tiny GPT-2（vocab=64、hidden=64、2 layers、2 heads）、8 条交错 prompt lengths `3,24,4,23,5,22,6,21`、`group_size=2`、`max_new_tokens=8`、count cap=8、length bucketing、20 iteration、1 warmup、3 repeats。GPU 0 在每组前确认无计算进程。唯一差异为是否配置 `rollout_max_padded_sequence_tokens=240`。

| 指标 | 无 budget | sequence budget=240 | 变化 / 解释 |
| --- | ---: | ---: | --- |
| prompt batches / prefill calls | 1 / 1 | 3 / 3 | 容量 guard 拆批 |
| decode forward calls | 7 | 21 | 三次 generate 的直接结果 |
| 单 batch prompt token 峰值 | 384 | 168 | -56.3% |
| 单 batch worst-case sequence token 峰值 | 512 | 232 | -54.7%，低于 240 |
| 累计 padded prompt tokens | 384 | 318 | -17.2% |
| rollout | 23.224 ms | 56.951 ms | +145.2% |
| train | 8.842 ms | 8.899 ms | 基本持平 |
| complete iteration | 32.487 ms | 66.286 ms | +104.0% |
| response token throughput | 3,770.739 tok/s | 1,863.139 tok/s | -50.6%* |
| allocator allocated peak | 27.59 MiB | 27.59 MiB | tiny 模型下持平 |

`*` response token 中位数为 122.5 / 123.5，采样 EOS 产生的 1 token 波动不能改变 wall-time 结论。

这组端到端证据把语义说清：sequence budget 是 capacity/admission 的可靠控制面，不是一个小模型吞吐优化。它把一次可能较大的生成请求拆成符合上界的请求；真实长上下文、大模型或显存临界服务应根据实际 KV bytes 和可用显存选择上限，而不是拿 tiny workload 的时延结果外推。

## 2026-08-19：capacity admission 与双 GPU one-step-lag prefetch 的组合

容量控制会把一次大 generate 拆成多个较小 generate，因此需要确认它既不破坏 trainer/rollout 独立副本的版本时序，也能与 prefetch 的安全一代 policy lag 组合。双 GPU benchmark 已接入变长 prompt、length bucketing 和两种 admission budget，并直接读取**实际完成 rollout** 的 `PromptBatchingStats`，而不是仅回显 CLI 配置。

固定两张 L20（trainer GPU 0、rollout GPU 1）、tiny GPT-2、8 条交错 prompt lengths `3,24,4,23,5,22,6,21`、`group_size=2`、`max_new_tokens=8`、count cap=8、length bucketing、sequence budget=240、20 个稳态 iteration、1 warmup、3 repeats。同步模式为 `rollout → reward → train → full state-dict sync`；prefetch 在独立 rollout replica 上先提交 `rollout(v_k)`，训练当前 batch 后等待该 future，再发布 `v_(k+1)`，故最多一代 policy lag。每组前确认 GPU 0/1 无计算进程。

| 指标 | 同步 | one-step-lag prefetch | 变化 |
| --- | ---: | ---: | ---: |
| 实际 prompt batches / iteration | 3 | 3 | 相同 admission work |
| 单 batch prompt token 峰值 | 168 | 168 | 相同 |
| 单 batch worst-case sequence token 峰值 | 232 | 232 | 均低于 240 |
| full next-rollout wall | 57.613 ms | 61.005 ms | 同量级 |
| prefetch overlap | 0 ms | 14.863 ms | rollout 的约 24.4% |
| rollout wait tail | 57.613 ms | 46.060 ms | 仍是关键路径 |
| train + reward | 9.288 ms | 14.861 ms | 可重叠窗口 |
| cross-device full state-dict sync | 1.257 ms | 1.271 ms | 持平 |
| iteration wall time | 68.319 ms | 62.346 ms | **-8.7%** |
| iteration throughput | 14.637 iter/s | 16.040 iter/s | +9.6% |
| response token throughput | 1,822.333 tok/s | 1,956.822 tok/s | +7.4%* |

`*` response token 中位数为 124.5 / 122.0，EOS 采样有轻微波动；调度收益应以 wall time 为主。两组 trainer/rollout allocator 峰值约 27.59 / 12.69 MiB，设备级 GPU 使用率分别约 2% / 10.7%（同步）和 2.1% / 12.1%（prefetch）；该 tiny workload 下这些低采样值只用于回归观察。

结论：capacity admission 与 safe policy-lag prefetch 可以组合，且版本/trajectory 契约不受影响；但它不是让容量保护“免费”的办法。由于三次 rollout generate 仍远长于 train，prefetch 只能隐藏训练窗口内约 14.9 ms，留下 46.1 ms tail。真实系统要同时追求容量安全和吞吐，下一步应是 continuous batching/paged KV 或更长的可重叠训练窗口，而不是无限降低静态 token budget。

## 2026-08-19：trainer-side length bucketing，一次逻辑 GRPO update 的梯度累积

此前训练侧 length bucketing 只存在于独立的固定 trajectory benchmark。`HuggingFaceTrainerWorker` 现在可配置 `train_micro_batch_size` / `train_max_padded_tokens`：先按完整 `prompt + response` 长度分桶，再逐个微批 forward/backward。每个微批的 GRPO loss 按其有效 response token 占全逻辑 batch 的比例缩放后累积梯度，最后**仅调用一次** `optimizer.step()`。因此这个路径降低 padding/峰值，而不改变“一次 Controller train = 一次策略更新”的语义。

CUDA 数值契约测试使用相同初始 1-layer GPT-2、无 dropout、一次 SGD update 和包含 1/2/3/4 response token 的 trajectory：整 batch 与 `micro_batch_size=2` / `max_padded_tokens=12` 的 length-bucketed accumulation 在 `loss`、`policy_loss` 和所有参数上满足严格数值容差；bucketed 路径的 padded sequence token 更低。

### 单卡端到端对照

固定 tiny GPT-2、8 条交错 prompt lengths `3,24,4,23,5,22,6,21`、`group_size=2`、`max_new_tokens=8`、rollout length bucketing、sequence budget=240、20 iteration、1 warmup、3 repeats。两组均只执行一次 optimizer step/iteration；唯一差异是训练端是否使用 `micro_batch_size=4`、`max_padded_tokens=128`。

| 指标 | 整 batch train | trainer bucketed accumulation | 变化 |
| --- | ---: | ---: | ---: |
| logical optimizer steps / iteration | 1 | 1 | 语义不变 |
| train micro-batches | 1 | 4 | 梯度累积 |
| train real sequence tokens | 339.5 | 339.0 | EOS 随机轻微波动 |
| train padded sequence tokens | 512 | 352 | -31.3% |
| train padding ratio | 33.69% | 3.69% | -32.0 pp |
| train wall time | 9.010 ms | 21.616 ms | +139.9% |
| iteration wall time | 67.225 ms | 79.306 ms | +18.0% |
| PyTorch allocator allocated peak | 27.59 MiB | 22.05 MiB | -20.1% |
| allocator reserved peak | 32.00 MiB | 28.00 MiB | -12.5% |

这个 tiny 模型的结论是一个有意保留的负/边界结果：减少 padding 不等于减少 wall time，因为分成四次 forward/backward 使 launch、Python 和 optimizer 固定开销占主导。它在显存受限时仍有价值；不要只看 token padding 便宣称端到端加速。

### 双 GPU prefetch 是否能回收额外训练时间

将同一 trainer bucket 配置接到 trainer GPU 0 / rollout GPU 1 的容量受限 pipeline 后：

| 指标 | 同步 | one-step-lag prefetch | 对照：无 trainer bucket 的 prefetch |
| --- | ---: | ---: | ---: |
| train micro-batches | 4 | 4 | 1 |
| train padding ratio | 3.98% | 4.00% | 约 33.7% |
| trainer allocated peak | 21.02 MiB | 21.02 MiB | 27.59 MiB |
| full next-rollout wall | 58.253 ms | 66.050 ms | 61.005 ms |
| prefetch overlap | 0 ms | 33.638 ms | 14.863 ms |
| rollout wait tail | 58.253 ms | 32.234 ms | 46.060 ms |
| iteration wall time | 81.973 ms | 67.421 ms | 62.346 ms |

分桶带来的更长 train 阶段确实将可隐藏 rollout 从约 14.9 ms 提高到 33.6 ms，并把同步到 prefetch 的同配置 iteration 降低 17.8%。但它仍比未分桶 prefetch 慢约 8.1%，所以正确的选择取决于约束：显存/容量优先时使用 trainer bucket；纯 tiny-workload 延迟优先时保持完整 batch。真实大模型的 kernel/FLOP 比不同，必须在目标硬件和长度分布上重测。
