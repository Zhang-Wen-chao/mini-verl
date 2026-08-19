# mini-verl 核心阅读地图

这个项目的目标不是缩写版的 Ray 或 vLLM，而是用最少代码解释一次 LLM
GRPO 更新必须保存、计算和校验什么。第一次阅读只需要下面四个文件：

```text
mini_verl/toy.py              可运行的完整训练（约 200 行）
mini_verl/protocol.py         rollout 和 learner 间不可变 Trajectory 契约
mini_verl/reward.py           rule reward 与组内 relative advantage
mini_verl/algorithms/grpo.py  masked clipped GRPO + reference KL
```

## 最小数据流

```text
categorical policy logits
  → sample G actions / record old logprobs
  → TrajectoryBatch(policy_version=k)
  → terminal rule reward
  → normalize rewards inside each prompt group
  → masked clipped GRPO loss
  → optimizer.step()
  → Controller advances policy_version to k + 1
```

`toy.py` 的 action 只是一个 token；这个替换让例子无需 tokenizer、下载模型或 GPU，
但没有跳过强化学习的关键语义：同一 prompt 的 G 个 rollout 共享一个 group，
`old_logprobs` 是采样策略，训练只能消费对应版本的轨迹，且 advantage 由组内奖励
归一化得到。

## 运行与验收

```bash
python -m pip install -e '.[torch]'
mini-verl-toy
python -m unittest discover -s tests -v
```

固定 seed 下，示例应从 `pass@1=0.125` 提升到 `1.000`。这证明的是数据契约和
GRPO 优化闭环，不是语言模型能力或生产吞吐。

## 读完小核后

其余模块是沿相同契约逐层加入的参考实现，而不是第二套算法：

| 目标 | 入口 | 增加的东西 |
| --- | --- | --- |
| 真正 CausalLM | `mini_verl/hf.py` | `generate`、response token 对齐、old/reference logprobs |
| 独立 rollout 副本 | `policy_sync.py`、`async_controller.py` | 状态复制和最多一代 policy lag |
| 变长 batch | `batching.py` | padding accounting 与 token admission |
| 训练恢复 | `checkpoint.py` | model、optimizer、RNG、policy version |
| 单机多卡 | `distributed.py`、`tests/ddp_*` | DDP 通信最小封装 |

上游官方 verl、Ray、vLLM、Qwen/GSM8K 系统实验不属于 `main` 的小核；它们在
`official-verl-grpo` 开发分支中单独对照。
