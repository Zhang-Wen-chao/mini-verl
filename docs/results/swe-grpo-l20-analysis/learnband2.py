#!/usr/bin/env python3
"""learnband.py 的补充:难度分层 + 「为什么是 797 组而不是 800」。

回答两个问题:
  1. 55 题按总体通过率分层长什么样?「已经会了」的题占多少?
  2. 200 个 rollout × 4 组/rollout = 800 才对吧 —— 797 这个数从哪来?

输入:SWE_DUMPS=<rollout_dumps 目录>(内含 rollout_*.pt)。见 README.md。
"""
import collections, glob, os, sys

import torch

D = os.environ.get("SWE_DUMPS", "")
if not os.path.isdir(D):
    sys.exit("需要 SWE_DUMPS=<内含 rollout_*.pt 的目录>,当前 = %r。见同目录 README.md。" % D)

groups = collections.defaultdict(list)
rolls = {}
for p in sorted(glob.glob(os.path.join(D, "rollout_*.pt")),
                key=lambda x: int(os.path.basename(x)[8:-3])):
    d = torch.load(p, map_location="cpu", weights_only=False)
    rid = d.get("rollout_id", -1)
    rolls[rid] = len(d["samples"])
    for s in d["samples"]:
        groups[(rid, s["label"])].append(float(s["reward"]))

print("=== 每个 rollout 的样本数分布 ===")
c = collections.Counter(rolls.values())
for k in sorted(c): print(f"  {k} 条样本的 rollout: {c[k]} 个")
print(f"  rollout 数 = {len(rolls)}, 样本总数 = {sum(rolls.values())}")
print(f"  (rollout,题) 组数 = {len(groups)}  ⇒ 平均每组 {sum(rolls.values())/len(groups):.2f} 个样本")
print()

tot = collections.defaultdict(lambda: [0, 0])
for (rid, lab), rs in groups.items():
    tot[lab][0] += int(sum(rs)); tot[lab][1] += len(rs)

def band(lo, hi):
    return sorted([l for l in tot if lo <= tot[l][0]/tot[l][1] < hi])

print("=== 按总体通过率分层(55 题) ===")
layers = [("0%(从未成功)",0,1e-9), ("0~10%",1e-9,0.10), ("10~30%",0.10,0.30),
          ("30~70%",0.30,0.70), ("70~90%",0.70,0.90), ("90~100%",0.90,1.0001)]
for name, lo, hi in layers:
    ls = band(lo, hi)
    print(f"  {name:<14} {len(ls):>2} 题")
    for l in ls:
        h, n = tot[l]
        if name in ("90~100%",):
            print(f"        {l:<42} {h}/{n} = {h/n:6.1%}")
print()
print("=== 「已经会了」的题(通过率 ≥90%)对梯度的贡献 ===")
easy = band(0.90, 1.0001)
print(f"  题数 {len(easy)}/{len(tot)} = {len(easy)/len(tot):.0%};"
      f"它们参与的组数 {sum(1 for (r,l) in groups if l in easy)}/{len(groups)}")

print()
print("=== 混合组数 ≤2/200 的「近乎死题」 ===")
near = [l for l in tot if 0 < tot[l][0]
        and sum(1 for (r,ll) in groups
                if ll==l and 0 < int(sum(groups[(r,ll)])) < len(groups[(r,ll)])) <= 2]
for l in sorted(near, key=lambda l: tot[l][1]):
    h, n = tot[l]
    print(f"  {l:<42} 总命中 {h}/{n} = {h/n:6.1%}")
