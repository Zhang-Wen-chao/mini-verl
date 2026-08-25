# slime Agent RL 训练记录(L20)

> 更新:2026-08-25
> 环境:slime 官方镜像容器 `slime-dev`(slimerl/slime:latest,Python 3.12, torch 2.11+cu129, sglang 0.5.15)
> 硬件:4×NVIDIA L20(48GB),GPU0 被其他租户占用,实际用 GPU 1-3
> 模型:Qwen3-4B-Instruct-2507(6.6GB,HF + Megatron torch_dist 双格式)
> 数据:dapo-math-17k(11MB)、aime-2024(40KB)、ReTool-SFT(4.9MB)
> 全部路径:`/mnt/storage01/zhangwenchao02/{models,data,experiments,retool-rl-smoke}`

## 里程碑 1:slime 训练闭环跑通(2026-08-25 05:13 UTC)

### 做了什么
1. 拉取 slime 官方镜像(daocloud 加速,42.3GB)并创建 `slime-dev` 容器(host 网络 + GPU + 挂载数据盘)
2. 下载 Qwen3-4B-Instruct-2507(HF_HUB_DISABLE_XET=1 解决 Xet 401 问题)
3. HF → Megatron torch_dist 权重转换(`convert_hf_to_torch_dist.py`)
4. 跑 retool 多轮工具 GRPO:20 rollout × 4 samples,GRPO + KL,训推分离

### 关键调试记录
- **colocate 模式 OOM**:4B 模型训练(38GB)+ 推理(9GB)挤同卡 → 44.5GB 不够,训练时 OOM
- **GPU0 被外部进程占用**:`bench_solo_gpu`(其他租户)占 43.8GB → 改用 GPU 1-3
- **训推分离解决**:`--tensor-model-parallel-size 2`(2 卡训练,每卡 ~20GB)+ 1 卡 SGLang 推理(31GB:模型 7.7 + KV 18.8)
- **max_tokens_per_gpu 4096 + mem_fraction 0.6** 控制显存

### 成功证据
- 模型真实调用工具:输出 `<tool_call>` → 沙箱执行 → 看到错误 → **自己修正重试** → 答对
- 沙箱安全生效:`Error: Import of 'gcd' is not allowed`、`SyntaxError`(代码块格式错被拦)
- 奖励信号:`score: 1.0, acc: True`(答对)与 `-0.9/-0.95`(答错惩罚)
- checkpoint 保存:`retool-rl-smoke/ckpt/iter_0000019`
- 20 个 rollout + 20 个训练更新全部完成,无 fatal 错误

### 训练数据观察
- response_len 从第 1 轮 mean 989 → 第 15 轮 mean 2063(模型多轮工具交互变长)
- prefix_cache_hit 从 0 → 提升(上下文复用)
- update_weights_time 0.27s(训推分离权重同步快)

## 下一步:4 种策略

见 `agent_rl_trends_notes.md` 的 4 个方向,在 slime 上的落地:
1. **在线反作弊**:retool tool_sandbox 已内置,对比有/无沙箱拦截
2. **验证器/奖励**:改 reward_func,过程奖励 vs 最终奖励
3. **GenAC 生成式 critic**:自定义 --custom-rm-path,LLM 先思考再打分
4. **OPID 技能自蒸馏**:slime 原生 --use-opd,需 teacher 模型

## 策略 1:在线反作弊对比(2026-08-25 完成)

### 设计
- 在 `tool_sandbox.py` 加 `SANDBOX_OFF` 环境变量开关:开=安全检查(现状),关=跳过 `_check_code_safety`
- 两组各 20 rollout × 4 samples,同一 retool GRPO,仅沙箱开关不同
- 代码:`experiments/strategy1/{tool_sandbox.py, sandbox_compare.sh}`
- 日志:`experiments/strategy1/{sandbox_on,sandbox_off}.log`

### 结果

| 指标 | 沙箱开(sandbox_on) | 沙箱关(sandbox_off) |
|---|---|---|
| 答对率 | 25%(10/40) | **32.5%(13/40)** |
| 平均奖励 | -0.525 | **-0.323** |
| 正奖励样本 | 10 | 13 |
| 沙箱拦截次数 | **13** | 2 |

### 结论
关沙箱后答对率 +7.5%、平均奖励 +0.2。原因:沙箱拦截了 13 次"不允许的导入"(如 `from math import gcd` 被白名单挡住),模型被迫绕路或失败;关沙箱后代码直接执行,解题更顺。

**验证了方向 2 的核心 trade-off**:反作弊不是免费的——过度拦截会误伤正常解题(`gcd` 这种无害导入),拖累 agent 学习。工业界(GLM-5.2)用"规则粗筛 + 大模型精判"避免一刀切,我们 20 rollout 复现了这个 trade-off。

### 备注
- 两组的 checkpoint 均保存(`retool-rl-smoke/{sandbox_on,sandbox_off}/ckpt/iter_0000019`)
- 关沙箱仍有 2 次拦截 = 内存/超时限制(非安全检查),符合预期
