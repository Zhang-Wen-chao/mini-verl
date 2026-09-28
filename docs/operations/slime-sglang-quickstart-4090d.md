# 在 4×RTX 4090D 上跑 slime / SGLang 官方 Quickstart

> 2026-09-26 实测。把 **SGLang 官方 quickstart** 和 **slime 官方 quickstart** 在
> 4090D 单机上各跑一遍，记录官方文档原文与实际执行的每一处偏离、实测数字、踩坑。
>
> **来源与引用说明**：本文档原本产生在 `slime` 仓库（内网 fork），后迁入本仓库。
> 文中多处按章节引用同一实验的另一份 runbook（`examples/swe_grpo/RUNBOOK_4090D.md`，
> 跑的是**打过 23 处补丁的 fork 副本**），**该文档未随本文档一并入库**，因此引用
> 指向的是外部材料而非本仓库路径。本份跑的是**镜像自带的官方代码**（`/root/slime`，
> HEAD `4c4adbbf`），因此能回答：那 23 处补丁里，哪些是 fork 陈旧、哪些是这台机器的
> 固有问题 —— 见 §6.1，结论是**两者都有，且官方代码当前仍有自己的 bug**。
>
> 内部主机名、IP、账号路径与容器名已按仓库脱敏约定替换为占位符：
> `<workdir>` 为测量机上的实验工作目录，`<container>` / `<old-container>` 为容器名。

## 0. 结论速览

| 项 | 状态 |
|---|---|
| SGLang quickstart（0.5B）| ✅ 四种请求方式全通，吞吐见 §2 |
| slime quickstart 链路（GRPO + dapo-math + Qwen3-1.7B）| ✅ 3 rollout + 3 训练步，job succeeded |
| 权重同步 | ✅ 9 次 `update_weights_from_tensor` 200 |
| 非零梯度 | ✅ `grad_norm` = 0.185 / 0.235 / 0.327 |
| rollout dump | ✅ `rollout_data/{0,1,2}.pt` + `train_data/{0,1}.pt` |
| **官方 quickstart 能否原样跑** | ❌ **不能**。两步必改：换模型、修一个官方代码里的 env 变量 bug |

**一句话**：官方 quickstart 在 4090D 上不是"配置调小"就能跑，而是**官方代码自身有一处
env 变量名错配**会让 TP 内存校验从"想关闭"变成"实际打开"，在有不均常驻进程的卡上必炸。
详见 §5.1 —— 这是本次唯一的硬发现，且**不是 fork 陈旧问题**。

## 1. 环境

```
Mac ──(跳板机)── 4090D 宿主机 (root)
  ├── 容器 <container>        ← 本次新建, 镜像 slimerl/slime:latest
  │     └── <workdir>/        ← 工作根
  │          ├── bin/   启动/检查脚本(见 §6)
  │          ├── data/  dapo-math-17k
  │          ├── models/ Qwen2.5-0.5B-Instruct
  │          ├── logs/
  │          └── runs/  Qwen3-1.7B_slime/ + dump/
  ├── 容器 <old-container>    ← 旧实验容器, 未改动
  └── host 上一个无关服务      ← 不相干, 未动
```

关键事实（沿用并验证另一份 runbook）：
- 4×RTX 4090D 24G，**无 NVLink**（`HAS_NVLINK=0`），驱动 550.127.05 / CUDA 12.4，sm89。
- 只有 `nerdctl`，无 docker CLI。
- GPU1/2 各有 612MiB、GPU3 有 4.3GiB 的 **70 天常驻进程**（非本项目，勿动）。
  GPU0 全空。
- 本地代理 `<local-proxy>` 可用（这次成了关键路径，见 §5.2）。
- 镜像内 SGLang `0.5.15.post1` / torch `2.11.0+cu129` / megatron-core `0.16.0rc0`。

## 2. Phase A — SGLang 官方 quickstart

官方路径：<https://docs.sglang.io/get-started/quickstart>。除 host 绑定外**逐条照抄**。

```bash
# 0) 建容器 + GeForce 必修: forward-compat libcuda 在 GeForce 上不支持
nerdctl pull docker.m.daocloud.io/slimerl/slime:latest      # 已在本地
nerdctl run -dt --name <container> --gpus all --net host --shm-size 64g \
  -v <host-home>:<host-home>:rw \
  -w <workdir> \
  docker.m.daocloud.io/slimerl/slime:latest /bin/bash
nerdctl exec <container> bash -c \
  "ln -sf libcuda.so.550.127.05 /usr/lib/x86_64-linux-gnu/libcuda.so.1"

# 1) 模型
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct

# 2) 起服务 (官方: --host 0.0.0.0; 共享机器改 127.0.0.1)
CUDA_VISIBLE_DEVICES=0 sglang serve --model-path models/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 30000
```

### 实测数字

| 项 | 值 |
|---|---|
| 模型下载（954M） | 51s |
| 冷启动到 ready | **72s**（二次启动 36s，page cache 命中） |
| 权重加载 | 0.98 GB / 0.35s |
| 服务态显存占用 | **22012 MiB**（GPU0）⚠️ 见下 |
| 吞吐 conc=1 | **231 tok/s** |
| 吞吐 conc=16 | **1739 tok/s** |
| `/docs` `/redoc` `/openapi.json` `/health` | 全 200 |
| 四种请求方式（cURL / OpenAI client / requests / 原生 `generate` 含 stream） | 全通 |

### 两个值得记的点

1. **`mem-fraction-static` 默认 0.9 会吃光 24G 卡**。0.5B 权重只占 0.98GB，
   但 server 起来后 GPU0 被占 **22 GB** —— 默认把 90% 显存预留成 KV cache。
   对独立 serving 无所谓，但**任何 colocate 场景都必须显式调小**。
2. **`model` 字段不做校验**。我们从本地目录起服务，请求里传 `"model": "models/Qwen2.5-0.5B-Instruct"`
   或任意串都返回 200；官方文档写的 `qwen/qwen2.5-0.5b-instruct` 只有在
   `--model-path` 用 HF repo id 时才字面一致。别把 200 当成"模型名对了"。

## 3. Phase B — slime 官方 quickstart（缩小版）

官方路径：`docs/en/get_started/quick_start.md` + 脚本骨架 `scripts/run-qwen3-4B.sh`。
**代码用镜像自带的 `/root/slime`（官方、未打补丁），不用 fork 副本。**

```bash
nerdctl exec <container> bash <workdir>/bin/run_qwen3_1.7b_dapo_4090d.sh
```

### 官方原文 vs 实际执行

| 项 | 官方 quickstart | 本次 | 原因 |
|---|---|---|---|
| 模型 | GLM-Z1-9B-0414 | **Qwen3-1.7B** | 9B 的 Adam 优化器状态（fp32 m+v+master ≈ 108G）在 4×24G=96G 上装不下 |
| docker 镜像 | `slimerl/slime:latest` | 同 | ✅ 照抄 |
| 数据集 | zhuzilin/dapo-math-17k | 同 | ✅ 照抄 |
| eval | aime-2024 | 未挂 | 冒烟省时 |
| `--num-rollout` | 3000 | **3** | 冒烟 |
| `--rollout-batch-size` | 32 | **8** | 显存/时长 |
| `--n-samples-per-prompt` | 8 | 8 | ✅ |
| `--global-batch-size` | 256 | **64** | = 8×8×1，满足官方约束 `rollout_batch × n_samples = global_batch × num_steps_per_rollout` |
| `--rollout-max-response-len` | 8192 | **4096** | KV 显存 |
| `--max-tokens-per-gpu` | 9216 | **4608** | 24G 卡 |
| `--sglang-mem-fraction-static` | 0.7 | **0.45** | 24G 卡 + colocate |
| TP / CP | 2 / 1 | 2 / 1 | ✅ 照抄 |
| `--rollout-num-gpus-per-engine` | 2 | 2 | ✅ 照抄 |
| GRPO / OPTIMIZER / MISC_ARGS | — | **逐字照抄** | grpo、kl-coef 0.00、eps-clip 0.2/0.28、adam 1e-6、attention-backend flash |
| 启动方式 | `ray start --head` + `ray job submit` + `train.py --colocate` | 同 | ✅ 照抄 |

### 实测数字

| 项 | 值 |
|---|---|
| 启动（ray + megatron + 2×SGLang 引擎 + 权重装载） | ≈ 3 min |
| 3 rollout + 3 训练步 全程 | **8m50s** |
| 权重同步 | 9 次 200 |
| decode 吞吐（2 引擎合计） | **~1500 → 1878 tok/s** 爬升 |
| 训练态显存 | 11–15 GB/卡（rollout 期），18–20 GB/卡（训练期） |

rollout 指标：

| rollout | raw_reward | truncated | 响应长度 |
|---|---|---|---|
| 0 | 0.031 | 0.969 | 4078 |
| 1 | 0.203 | 0.797 | 3773 |
| 2 | 0.406 | 0.594 | 3583 |

训练步：

| step | loss | grad_norm | kl_loss | entropy |
|---|---|---|---|---|
| 0 | 0.0 | **0.185** | 0.0 | 0.310 |
| 1 | 0.0 | **0.235** | 0.00094 | 0.279 |
| 2 | 9.3e-10 | **0.327** | 0.00079 | 0.219 |

### dump 证据（验收）

`runs/dump/rollout_data/2.pt` → `{rollout_id: 2, samples: list[64]}`，单样本字段齐全：
`prompt / tokens[4199] / response_length 4096 / label / reward / loss_mask[4096] / weight_versions ['3'] / status 'truncated' / prefix_cache_info`。

**64 样本 reward 呈 `[0,1,1,1,0,1,1,1,0,0,0,0,0,...]` 的 0/1 混合** → 组内有方差 →
advantage 非零 → `grad_norm` 非零。**精确命中另一份 runbook §4.23 的健康判据**
（全 0 组是 GRPO 数学预期，混合组必须有梯度）。

`prefix_cache_info {'cached_tokens': 99, 'total_prompt_tokens': 100}` —— RadixAttention 前缀缓存在工作。

## 4. ⚠️ 不要过度解读 reward 上升

raw_reward 0.031 → 0.203 → 0.406 看着像"在学"，**但这 3 步不足以支撑该结论**：

- 它和 `truncated` 下降（0.969 → 0.797 → 0.594）**同步**。截断样本判 reward=0，
  所以截断率一降，reward 必然升 —— 驱动量是"生成长度分布"，不一定是"答对率"。
- 只跑了 3 个 rollout，`--rollout-shuffle` 每轮换题，题目难度本身在波动。
- lr=1e-6、3 步，参数几乎没动。

**要判断"是否在学"，至少需要**：固定题集 + 关 shuffle + 跑几十步 + 看 aime eval 曲线。
本文档不下这个结论。

> 这条与官方 verl quickstart 的判读是同一族问题（那份 runlog 在**另一条分支**
> `verl-quickstart-4090d` 的 `official_verl/docs/runlogs/2026-09-26-4090d-verl-quickstart-ppo-gsm8k.md`，
> 尚未在本分支，故此处不写链接）：那边是硬 `max_response_length` 让**答案写不完**
> （响应长度 315→170、cap ratio 0.152→0.012，曲线主要由"答案变得可解析"驱动）；
> 这边是截断率把 reward 抬起来。共同点：**被度量的是长度/格式，不是能力。**

## 5. 问题清单

### 5.1 ⚠️ 核心发现：官方代码的 TP 内存校验 env 名错配

**症状**：SGLang 引擎启动即死，surface 报错很误导 ——
```
Exception: Server process terminated unexpectedly.
  File "/root/slime/slime/backends/sglang_utils/sglang_engine.py", line 83, in _wait_server_healthy
```
真正的死因被埋在 Ray worker 日志里（`_wait_server_healthy` 只轮询 `p.is_alive()`，吞掉了子进程异常）：
```
RuntimeError: The memory capacity is unbalanced. Some GPUs may be occupied by other processes.
pre_model_load_memory=18.88, local_gpu_memory=22.53, local_gpu_memory*0.9=20.28
```

**根因链**（三环，全部确认）：

| # | 位置 | 内容 |
|---|---|---|
| 1 | slime `slime/backends/sglang_utils/engine_group.py:129-130` | 想**关闭**校验，于是设 `SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=true` 和 `SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=true` |
| 2 | sglang `srt/environ.py:283` | 现在读的是 **`SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK`**（默认 `True`）。**两个 `DISABLE` 名字都不在其读取列表里** |
| 3 | sglang `srt/environ.py:997` | deprecated shim 是 `os.environ[new_name] = os.environ[old_name]` —— **盲拷值、不反转语义**。于是 `SGL_DISABLE...=true` 被拷成 `SGLANG_ENABLE...=true`，**把校验打开了** |

**结论**：slime 那句"关闭校验"实际是"打开校验"。GPU3 有 4.3G 常驻 → 可用显存 18.88G
< 总显存 90%（20.28G）→ 必炸。**只要 TP 组内各卡空闲显存不均就复现**，
与 fork 无关，官方最新代码 `4c4adbbf` 原样存在。

**修法（本次采用）**：必须从**源头**改，否则会被 shim 覆盖回去 ——
```
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=false
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false
```
（同时写进 `ray job submit --runtime-env-json` 的 `env_vars`。）

只设 `SGLANG_ENABLE_...=false` **不够**：shim 会用 `SGL_DISABLE...=true` 把它覆盖掉。

**建议的上游修法**：`engine_group.py` 里改为设 `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK`（正确名字），
或 sglang 侧把 shim 改成会反转语义的映射。当前写法在任何一台有不均常驻进程的机器上都是雷。

**验证**：改后同一条消息降级为 warning，引擎正常起，训练跑完。

### 5.2 hf-mirror 下不动大模型权重（本机必读）

**症状**：`hf download` 小文件（tokenizer）秒下，`model.safetensors` 卡死/超时。

**原因**：hf-mirror 把权重 **302 重定向到 `cas-bridge.xethub.hf.co`（Xet 存储）**，
该域名**不在镜像范围内**。`HF_HUB_DISABLE_XET=1` 挡不住服务端的 302。

**修**：走代理直连真 HF。
```bash
export HTTPS_PROXY=http://<local-proxy> HTTP_PROXY=http://<local-proxy>
export HF_HUB_DISABLE_XET=1
unset HF_ENDPOINT        # 不要用 hf-mirror
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```
实测 954M / 51s。

> 注意：另一份 runbook §2.5 用 hf-mirror 成功了（Qwen3-1.7B）。说明这是**按模型而定**的
> —— 取决于该 repo 的权重是否走 Xet。**遇到大文件卡住先怀疑这条**。
> 数据集（dapo-math-17k）走 hf-mirror 没问题（4.5s）。
>
> 同一天在官方 verl quickstart 上**再次复现**（Qwen2.5-0.5B-Instruct 同样中招），
> 说明这条与训练框架无关，是这一侧的属性。

### 5.3 未复现的老坑（说明官方已处理）

| 老 runbook | 本次结果 |
|---|---|
| §4.9 breakable CUDA graph × memory saver | **未复现**。官方 `engine_group.py` 已设 `SGLANG_MEMORY_SAVER_CUDA_GRAPH: "true"`，日志里 colocate 期 `cuda graph: False`，正常 |
| §4.11 TP 内存不均 | **复现了**，但根因定位比上次更深（见 §5.1） |
| §4.2 GeForce forward-compat libcuda | **复现**，一条 symlink 修掉 |
| §4.8 slime×sglang 参数名不兼容 | **未复现**（官方代码已同步 sglang 0.5.15） |

## 6. 收获

1. **官方代码的坑和 fork 的坑是两类**。另一份的 23 条里，参数名不兼容（§4.8）、
   CUDA graph（§4.9）等是 **fork 陈旧**；而 libcuda forward-compat、TP 内存校验
   是**机器固有问题**，官方代码一样中招 —— 且 §5.1 那个 env 名错配是官方代码**当前**的 bug，
   不是 fork 的问题。**"换官方代码就好了"这个假设是错的**。
2. **surface 报错会骗人**。`Server process terminated unexpectedly` 什么也没说，
   真因在 Ray worker 日志。slime 的 `_wait_server_healthy` 只查 `p.is_alive()`，
   子进程的 Python 异常被吞掉。**排查这类问题的第一步是去 `/tmp/ray/session_latest/logs/worker-*.err` 捞 traceback**。
3. **SGLang 默认 mem-fraction 0.9** 对 colocate 是危险的，必须显式设。
4. **`--sglang-` 透传实操**：`--sglang-mem-fraction-static 0.45` 直接落到引擎的
   `ServerArgs`，不需要改 slime 代码 —— 官方 quickstart 文档里那句话是真的。
5. **官方默认拓扑是 2 引擎 × TP2 + sgl-router**，与另一份 fork 实验用的 TP4 单引擎不同。
   本次 2 引擎合计 decode ~1500-1878 tok/s（1.7B，2×24G 推理）。
6. **dapo-math + 1.7B 能出真信号**：64 样本里 0/1 混合，`grad_norm` 非零。
   比 SWE-GRPO 实验（1.7B 在 SWE-bench 上恒 0；记录在 `swe-grpo-l20` 分支的
   `docs/results/swe-grpo-l20-negative-result.md`，未在本分支）干净得多 ——
   **做"链路是否健康"的冒烟，数学题比 SWE 题好得多**（不需要沙箱、不需要镜像、reward 立即可判）。
7. **1.7B 的 thinking 长度是真的长**：`--rollout-max-response-len 4096` 下
   截断率 59%–97%，响应长度贴着上限（3583–4078）。做真实训练必须放大，
   但 24G 卡的 KV 预算撑不住 —— 这是 4090D 上 1.7B 做数学 RL 的**主要瓶颈**，不是算力。

## 7. 已知限制 / 下一步

1. 只跑 3 个 rollout，**不能得出任何"模型变好/变坏"的结论**（见 §4）。
2. `--sglang-mem-fraction-static 0.45`、`--max-tokens-per-gpu 4608` 是**保守猜测值**，
   未经扫描；可能还有余量。
3. Qwen3-4B（官方 quickstart 点名的等价模型）**未验证** —— 另一份 runbook §0 说它是
   "4090D 上可行的最大"，本份未证实。要做：换 4B 复跑本脚本，观察显存与截断率。
4. aime-2024 eval 未挂。加上后能看真实准确率曲线。
5. 容器 `<container>` 保留（供 3/4 的实验），`ray stop` 已执行、端口已释放。
6. 所有脚本在测量机 `<workdir>/bin/`，**未随本文档入库**（本仓库 `docs/` 只放文档，
   且尚无 slime 可执行脚本的位置）。

## 8. 复现步骤（最小）

```bash
# 前提: 容器 <container> 已建、libcuda 已修、端口 6379/8265 空闲
ssh <4090d-host>

# ---- Phase A ----
nerdctl exec -d <container> bash -c "cd <workdir> && \
  CUDA_VISIBLE_DEVICES=0 nohup sglang serve --model-path models/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 30000 > logs/sglang_serve.log 2>&1 &"
# 等 "fired up and ready to roll"
nerdctl exec <container> bash <workdir>/bin/phase_a.sh
nerdctl exec <container> bash -c "pkill -9 -f sglang"

# ---- Phase B ----
nerdctl exec -d <container> bash -c "cd <workdir> && \
  nohup setsid bash bin/run_qwen3_1.7b_dapo_4090d.sh > logs/run_dapo_1.7b.log 2>&1 &"

# 预期: "Job 'raysubmit_xxx' succeeded"
#       grad_norm 非零, runs/dump/rollout_data/{0,1,2}.pt 存在
nerdctl exec <container> python3 bin/inspect_dump.py
```

诊断入口（出问题时先看这里，不要只看 surface 报错）：
```bash
nerdctl exec <container> bash -c \
  'grep -rlniE "Traceback|Error" /tmp/ray/session_latest/logs/worker-*.err | head'
```
