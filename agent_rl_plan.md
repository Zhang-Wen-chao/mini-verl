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
2. ⏳ 读 gsm8k_tool_agent_loop.py 数据脚本，生成工具调用训练数据
3. ⏳ 找 verl 的 agent 训练入口（main_ppo + agent 配置 / 或教程示例）
4. ⏳ 用 Qwen3-0.6B 跑通 agent GRPO smoke（几十步）
5. ⏳ 换 Qwen3.5-4B 跑正式 agent RL
6. ⏳ 评测 + 记录 + 提交

## 教训（追加到 lessons-learned）

- 选框架前先确认其后端依赖与现有环境的兼容性；
  AgentRL 的 `[vllm]` extra 是**声明依赖但无实际 worker** 的"幌子"，训练路径只有 sglang。
- sglang[all] 依赖树庞大且版本强绑定（torch==2.7.1 + flashinfer 冲突），
  在国内网络下安装成本极高。优先选已验证环境的方案。
- 大文件下载用 wget（断点续传+重试），别用 pip 直接下（会永久挂起）。
