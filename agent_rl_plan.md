# agent RL 实验规划（2026-08-20）

分支：`agent-rl`（worktree：`/Users/bilibili/Documents/mini-verl-agent-rl`）

## 目标

在 L20（4×L20）上跑通一个多轮工具调用的 agentic RL 训练，
复现"模型学会调用工具解决任务"的 GRPO 流程。

## 方案选型（重要决策记录）

### 尝试过：AgentRL（THUDM）+ sglang 后端 → 放弃

- clone 了 `THUDM/AgentRL`，理解了架构（trainer/controller/worker、GRPO、异步流水线）。
- **卡点**：sglang 后端强制 `torch==2.7.1` + flashinfer 源码编译，且
  `sglang[all]==0.4.8` 与 `flashinfer-python==0.2.6.post1` 存在**依赖冲突**
  （sglang 要 torch 2.7.1，flashinfer 要 torch 2.13.*），pip 无法解析。
- 试了 4 次安装：PyPI 直连慢（43KB/s）→ 阿里云镜像（torch 821MB 下载挂起）→
  清华源（flashinfer 编译）→ 本地 torch wheel + 清华源（依赖冲突 ResolutionImpossible）。
- 时间成本高、边际收益低，**果断止损**。

### 改用：verl 官方 agent_loop（推荐，已确认可行）

- **verl `experimental/agent_loop` 已内置多轮工具 RL**：
  - `tool_agent_loop.py`：多轮工具调用循环（AgentState 状态机）
  - `tool_parser.py`：工具调用解析
  - 现成数据脚本：`gsm8k_tool_agent_loop.py`、`gsm8k_multiturn_w_tool.py`、
    `aime2024_multiturn_w_tool.py`（多轮 + 工具）
  - 教程：`examples/tutorial/agent_loop_get_started/agent_loop_tutorial.ipynb`
- **环境零成本**：复用已验证的 verl venv（vllm 0.24 + torch 2.11 + FSDP2），
  不需要 sglang/flashinfer。
- 与 679 步数学 GRPO 同一套 verl 生态，迁移顺畅。

## 落地步骤

1. ✅ 确认 verl 有 agent_loop 支持
2. ✅ 评估官方 agent-loop 路径与现有环境，确认采用 Strategy 2 的 retool 工具协议
3. ✅ 完成多轮 rollout、答案解析、tool observation 隔离和 recovery prompt 的评测协议
4. ✅ 用 Qwen3-0.6B 完成 agent GRPO smoke 与早期 reward/stability 对照
5. ✅ 用 Qwen3-4B 完成 Strategy 2 的 outcome/process/quality-process reward 训练与 checkpoint staging
6. ✅ 完成 8K 四策略、16K 容量消融和修复协议后的 v3 paired held-out 评测，结果记录在 [`experiments/strategy2/final_eval_report.md`](experiments/strategy2/final_eval_report.md)

## 最终状态（2026-08-27）

本分支的 Strategy 2 follow-up 已完成。远端四张 L20 已释放，所有评测 arm 均有 30 条逐题记录和汇总文件。最终没有统计显著的 reward winner；v2 的主要问题是工具调用和无效动作增加、答案终止率下降，v3 修复协议后只出现未经显著性验证的净增 1 题信号。后续若继续，优先做独立 seed 和更大 held-out，而不是直接延长当前 checkpoint 的训练。

## 教训（追加到 lessons-learned）

- 选框架前先确认其后端依赖与现有环境的兼容性；
  AgentRL 的 `[vllm]` extra 是**声明依赖但无实际 worker** 的"幌子"，训练路径只有 sglang。
- sglang[all] 依赖树庞大且版本强绑定（torch==2.7.1 + flashinfer 冲突），
  在国内网络下安装成本极高。优先选已验证环境的方案。
- 大文件下载用 wget（断点续传+重试），别用 pip 直接下（会永久挂起）。
