#!/usr/bin/env python3
"""关键:dump 里重复的条目是「同一采样的副本」还是「两次不同的采样」?

前者 ⇒ 去重后比率不变;后者 ⇒ 文档里的训练池曲线可能被加权偏了。

**这一步不足以裁决。** 本脚本只能说明重复 index 的奖励是否一致(实测:全一致,
0 处不一致 —— 形态上像纯副本)。真正的裁决在 roll_compare.py:拿训练器自己日志里的
`rollout/raw_reward` 逐 rollout 对照,看「原样」还是「去重」才是训练器实际吃进去的批次。

结论(实测):**原样**才是。去重会把整条曲线整体抬高约 3.5pp,而曲线形状照样单调好看 ——
没有任何内部信号提示它错了。详见 README.md。

输入:SWE_DUMPS=<rollout_dumps 目录>(内含 rollout_*.pt)。
"""
import collections, glob, os, sys

import torch

D = os.environ.get("SWE_DUMPS", "")
if not os.path.isdir(D):
    sys.exit("需要 SWE_DUMPS=<内含 rollout_*.pt 的目录>,当前 = %r。见同目录 README.md。" % D)

same = diff = 0
raw = collections.defaultdict(list)   # (rid, lab) -> [(idx, reward)]
for p in sorted(glob.glob(os.path.join(D, "rollout_*.pt")),
                key=lambda x: int(os.path.basename(x)[8:-3])):
    d = torch.load(p, map_location="cpu", weights_only=False)
    rid = d.get("rollout_id", -1)
    for s in d["samples"]:
        raw[(rid, s["label"])].append((s["index"], float(s["reward"])))

for k, lst in raw.items():
    by = collections.defaultdict(set)
    for ix, r in lst:
        by[ix].add(r)
    for ix, rs in by.items():
        if len(rs) == 1:
            same += 1
        else:
            diff += 1

print("=== 重复 index 的奖励是否一致 ===")
print(f"  一致(纯副本)的 (组,index) 对: {same}")
print(f"  不一致(两次不同采样)的对 : {diff}")
print()

# 去重后按步分层
def bucket(rid): return rid // 20 * 20

agg = collections.defaultdict(lambda: [0, 0])       # 原样(不去重)
ded = collections.defaultdict(lambda: [0, 0])       # 去重
seen = set()
for (rid, lab), lst in raw.items():
    b = bucket(rid)
    for ix, r in lst:
        agg[b][0] += int(r); agg[b][1] += 1
        key = (rid, lab, ix)
        if key in seen: continue
        seen.add(key)
        ded[b][0] += int(r); ded[b][1] += 1

print("=== 训练池 reward=1 比率:原样(文档口径) vs 去重 ===")
print(f"{'步数':>8} {'原样':>20} {'去重':>20}")
for b in sorted(agg):
    a, dn = agg[b], ded[b]
    print(f"{b:>3}-{b+19:<4} {a[0]:>5}/{a[1]:<6}={a[0]/a[1]:6.2%}   {dn[0]:>5}/{dn[1]:<6}={dn[0]/dn[1]:6.2%}")

print()
print("=== 50 步等宽聚合(按 rollout_id // 50 分桶) ===")
# 注意:桶边界必须按 rollout_id 直接切,不能用「三块 20 步桶求和」去凑 50 ——
# 20 步桶的起点是 0/20/40/60…,取 (0,20,40) 实际覆盖 0–59,标签却是 0–49。
a50 = collections.defaultdict(lambda: [0, 0])
d50 = collections.defaultdict(lambda: [0, 0])
seen50 = set()
for (rid, lab), lst in raw.items():
    b = rid // 50 * 50
    for ix, r in lst:
        a50[b][0] += int(r); a50[b][1] += 1
        key = (rid, lab, ix)
        if key in seen50: continue
        seen50.add(key)
        d50[b][0] += int(r); d50[b][1] += 1
for b in sorted(a50):
    a, dn = a50[b], d50[b]
    print(f"  {b:>3}-{b+49:<3} 原样 {a[0]:>4}/{a[1]:<5}={a[0]/a[1]:6.2%}   去重 {dn[0]:>4}/{dn[1]:<5}={dn[0]/dn[1]:6.2%}")
