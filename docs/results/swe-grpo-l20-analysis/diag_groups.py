#!/usr/bin/env python3
"""诊断:为什么 (rollout,题) 组的样本数不都等于 8?

背景:按声明的协议应当是 200 rollout × 4 组 × 8 样本 = 6400 条。实际数出来的组
不止 797 个组的算术,且组内样本数会超过 8 —— dump 的 `samples` 列表里有**条目级重复**
(同一个 (rollout, label, index) 出现多次)。

本脚本给出重复的规模与形态。**注意它只描述现象,不裁决该不该去重** ——
该不该去重要看训练器自己吃了什么,那是 roll_compare.py 的活(见 README.md)。

输入:SWE_DUMPS=<rollout_dumps 目录>(内含 rollout_*.pt)。
"""
import collections, glob, os, sys

import torch

D = os.environ.get("SWE_DUMPS", "")
if not os.path.isdir(D):
    sys.exit("需要 SWE_DUMPS=<内含 rollout_*.pt 的目录>,当前 = %r。见同目录 README.md。" % D)

size_hist = collections.Counter()
dup_ix = 0
grp_with_dup = 0
per_dump_groups = collections.Counter()
detail = []
for p in sorted(glob.glob(os.path.join(D, "rollout_*.pt")),
                key=lambda x: int(os.path.basename(x)[8:-3])):
    d = torch.load(p, map_location="cpu", weights_only=False)
    rid = d.get("rollout_id", -1)
    by = collections.defaultdict(list)
    for s in d["samples"]:
        by[s["label"]].append(s)
    per_dump_groups[len(by)] += 1
    for lab, ss in by.items():
        size_hist[len(ss)] += 1
        ixs = [s["index"] for s in ss]
        if len(set(ixs)) != len(ixs):
            grp_with_dup += 1
            dup_ix += len(ixs) - len(set(ixs))
        if len(ss) > 8 and len(detail) < 12:
            detail.append((rid, lab, len(ss), sorted(set(ixs))[:20], len(set(ixs))))

print("=== 每个 (rollout,题) 组的样本数直方图 ===")
for k in sorted(size_hist):
    print(f"  {k:>4} 个样本的组: {size_hist[k]}")
print()
print("=== 每个 rollout 含几道题 ===")
for k in sorted(per_dump_groups):
    print(f"  {k} 道题的 rollout: {per_dump_groups[k]}")
print()
print(f"=== index 有重复的组: {grp_with_dup} 个,重复 index 共 {dup_ix} 个 ===")
print("=== 样本数 >8 的组的样例(rollout, 题, n, 不同 index, 不同 index 数) ===")
for rid, lab, n, uix, nu in detail:
    print(f"  rollout {rid:>3} {lab:<34} n={n:<4} uniq_index={nu:<3} {uix}")
