#!/usr/bin/env python3
"""55 题训练池的「可学带」量化:从 200 个 rollout dump 逐条重算。

只回答一个问题:**训练池里哪些题真的贡献了梯度?**

GRPO 的组内 advantage:一组 8 个样本里 reward 全 0 或全 1 ⇒ advantage 恒 0 ⇒ 无梯度。
只有 1..7/8 的「混合组」才产生梯度。所以「可学带」= 混合组的出现率。

用它能否掉一个很自然的反驳:「训练池曲线能涨,是因为死题太多、只能背」
—— 实测 55 题里 44 题(80%)都出现过混合组。

输入:SWE_DUMPS=<rollout_dumps 目录>(内含 rollout_*.pt)。见 README.md。
"""
import collections, glob, os, sys

import torch

D = os.environ.get("SWE_DUMPS", "")
if not os.path.isdir(D):
    sys.exit("需要 SWE_DUMPS=<内含 rollout_*.pt 的目录>,当前 = %r。见同目录 README.md。" % D)

# (rollout_id, label) -> [rewards]
groups = collections.defaultdict(list)
n_dumps = 0
for p in sorted(glob.glob(os.path.join(D, "rollout_*.pt")),
                key=lambda x: int(os.path.basename(x)[8:-3])):
    d = torch.load(p, map_location="cpu", weights_only=False)
    rid = d.get("rollout_id", -1)
    n_dumps += 1
    for s in d["samples"]:
        groups[(rid, s["label"])].append(float(s["reward"]))

labels = sorted({lab for (_, lab) in groups})
n_groups = len(groups)

# 每组分类
cls = collections.Counter()
per_label = collections.defaultdict(lambda: collections.Counter())
per_label_total = collections.defaultdict(lambda: [0, 0])  # [命中, 样本]
for (rid, lab), rs in groups.items():
    n = len(rs); k = int(sum(rs))
    per_label_total[lab][0] += k; per_label_total[lab][1] += n
    if n == 0:            c = "空"
    elif k == 0:          c = "全败"
    elif k == n:          c = "全胜"
    else:                 c = "混合(有梯度)"
    cls[c] += 1
    per_label[lab][c] += 1

print(f"dump 数 = {n_dumps}   (rollout,题) 组数 = {n_groups}   题数 = {len(labels)}")
print()
print("=== A. 组一级(每一组 = 一道题在一次 rollout 里的 8 个样本) ===")
for c in ("全败", "混合(有梯度)", "全胜", "空"):
    if cls[c]:
        print(f"  {c:<14} {cls[c]:>5} 组  ({cls[c]/n_groups:6.1%})")
grad = cls["混合(有梯度)"]
print(f"  ⇒ 真正产生梯度的组只有 {grad}/{n_groups} = {grad/n_groups:.1%}\n")

print("=== B. 题一级(把 200 个 rollout 合成一道题的总体) ===")
bucket = collections.Counter()
for lab in labels:
    hit, tot = per_label_total[lab]
    if hit == 0:                 b = "从未成功(全 200 组全败)"
    elif hit == tot:             b = "永远成功"
    elif per_label[lab]["混合(有梯度)"] == 0: b = "有成功但从未出现混合组"
    else:                        b = "出现过混合组(可学带)"
    bucket[b] += 1
for b, n in bucket.most_common():
    print(f"  {b:<28} {n:>3} 题  ({n/len(labels):6.1%})")
print()

print("=== C. 可学带逐题(出现过混合组的题) ===")
band = [lab for lab in labels if per_label[lab]["混合(有梯度)"] > 0]
band.sort(key=lambda l: -per_label[l]["混合(有梯度)"])
for lab in band:
    hit, tot = per_label_total[lab]
    m = per_label[lab]["混合(有梯度)"]
    print(f"  {lab:<44} 混合组 {m:>3}/200  总命中 {hit:>4}/{tot:<4} = {hit/tot:6.1%}")
print()
print("=== D. 从未成功过的题数(对梯度零贡献) ===")
dead = [lab for lab in labels if per_label_total[lab][0] == 0]
print(f"  {len(dead)}/{len(labels)} 题 = {len(dead)/len(labels):.1%}")
