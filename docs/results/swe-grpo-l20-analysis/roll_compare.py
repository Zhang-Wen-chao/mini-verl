#!/usr/bin/env python3
"""裁决脚本:逐 rollout 对照「训练器自报 raw_reward」 vs 「我从未去重/去重 dump 算的比率」。

为什么必须有它:分析脚本自己算出来的东西**不能自己给自己当判据**。
dedup_curve.py 里「去重更干净」有两处内部证据支持(符合直觉;去重后每个 20 步桶
恰好 640 = 20×4×8,与声明协议逐字吻合)—— 差一点就把「文档曲线错了」写进去。
击穿它的是**训练器自己日志里的 `rollout/raw_reward`**(每批样本级平均 reward),
那是被分析系统**之外**、但**同源**的数。

只有逐点对照才排除桶均值互相抵消的假象:均值会互相抵消出假吻合。

实测结果:原样 与训练器 61 个 rollout 差 0.00%(其中 60/61 逐点一致);
去重则系统性偏高 +3.5pp ⇒ **那批重复就是训练器真正吃进去的批次。**

⚠️ 已知离群:rollout 10 上训练器 35.59% vs dump 16.98%。**不是 dump 的缺陷**,
是我自己取数的谱系问题 —— 这 61 个 raw_reward 来自三段不同的 run:
attempt1.log 覆盖 0–9、attempt2.log 只有 10、主日志 50–99。提取混淆,不是数据缺陷。

输入:
  SWE_DUMPS=<rollout_dumps 目录>
  SWE_LOGS =<训练日志,冒号分隔>(默认取 SWE_DUMPS 同级的 *run*.log)
"""
import collections, glob, os, re, statistics as st, sys

import torch

D = os.environ.get("SWE_DUMPS", "")
if not os.path.isdir(D):
    sys.exit("需要 SWE_DUMPS=<内含 rollout_*.pt 的目录>,当前 = %r。见同目录 README.md。" % D)

LOGS = [p for p in os.environ.get("SWE_LOGS", "").split(":") if p]
if not LOGS:
    LOGS = sorted(glob.glob(os.path.join(os.path.dirname(D.rstrip("/")), "**", "*.log"),
                            recursive=True))
    if not LOGS:
        sys.exit("需要 SWE_LOGS=<训练日志,冒号分隔>(含 'rollout <i>: {...rollout/raw_reward...}' 行)。")

# 1) 训练器自报
tr = {}
pat = re.compile(r"rollout (\d+): \{[^\n]*?'rollout/raw_reward': ([0-9.eE+-]+)")
for lg in LOGS:
    if not os.path.exists(lg):
        print(f"  (跳过不存在的日志: {lg})", file=sys.stderr); continue
    with open(lg, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                tr.setdefault(int(m.group(1)), float(m.group(2)))
print(f"训练器自报 raw_reward 的 rollout 数 = {len(tr)}  (min {min(tr) if tr else '-'}, max {max(tr) if tr else '-'})")

# 2) dump 两种口径
raw = collections.defaultdict(lambda: [0, 0])
ded = collections.defaultdict(lambda: [0, 0])
seen = set()
for p in sorted(glob.glob(os.path.join(D, "rollout_*.pt")),
                key=lambda x: int(os.path.basename(x)[8:-3])):
    d = torch.load(p, map_location="cpu", weights_only=False)
    rid = d.get("rollout_id", -1)
    for s in d["samples"]:
        r = float(s["reward"])
        raw[rid][0] += int(r); raw[rid][1] += 1
        k = (rid, s["label"], s["index"])
        if k in seen: continue
        seen.add(k)
        ded[rid][0] += int(r); ded[rid][1] += 1

both = sorted(set(tr) & set(raw))
print(f"两边都有的 rollout = {len(both)}")
print(f"{'rollout':>7} {'训练器':>9} {'dump原样':>9} {'dump去重':>9}   {'原样-训练器':>10} {'去重-训练器':>10}")
d_raw = []; d_ded = []
for r in both:
    a = raw[r][0]/raw[r][1]; b = ded[r][0]/ded[r][1]
    d_raw.append(a - tr[r]); d_ded.append(b - tr[r])
    if r < 12 or r % 25 == 0:
        print(f"{r:>7} {tr[r]:9.2%} {a:9.2%} {b:9.2%}   {a-tr[r]:+10.2%} {b-tr[r]:+10.2%}")
print()
print(f"原样  与训练器的平均差 = {st.mean(d_raw):+.2%}   绝对差均值 = {st.mean(map(abs,d_raw)):.2%}")
print(f"去重  与训练器的平均差 = {st.mean(d_ded):+.2%}   绝对差均值 = {st.mean(map(abs,d_ded)):.2%}")
