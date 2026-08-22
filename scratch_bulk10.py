#!/usr/bin/env python3
"""把 kv 教师从 4 篇扩到 **10 篇**（cat 语料硬上限）后重估三件事。

为什么值得做：现在下场的那张 kv 表（Retr.KV@0.1 **+29.40★**）**只由 4 篇拟出**，
而跨篇信度也只有 **6 对**的估计（+0.1630）。补到 10 篇后 **45 对**，
估计精度提升一个量级。

**判据先写死**：
  ① 10 篇的 CV R² 与 4 篇比 —— 升 ⇒ 数据量确实受限；持平 ⇒ 4 篇已够；
  ② 跨篇两两 Spearman（45 对）与 4 篇的 6 对比 —— 这是**静态表能成立的前提**；
  ③ `Spearman(u_4篇, u_10篇)` —— 高 ⇒ 方向稳定、那 +29.40 站在可复现的方向上；
     低 ⇒ **表随标定集漂移**，+29.40 的可复现性存疑（那会是个重要的负结果）。
"""
import json
import sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/home/ubuntu/zxy/vlm-memory")
from scratch_psyn_downstream import _fit                      # noqa: E402

A = json.load(open("scratch_adv_grad_bulk.json"))
B = json.load(open("scratch_adv_grad_bulk2.json"))
ALL = A + B
print(f"4 篇 {len(A)} 条 ・ 补篇 {len(B)} 条 ・ 合计 {len(ALL)} 条 / "
      f"{len({x['doc'] for x in ALL})} 篇")

r4, l4, u4 = _fit(A)
rA, lA, uA = _fit(ALL)
print(f"\n① CV R²：4 篇 {r4:+.4f}（λ={l4:g}） → 10 篇 {rA:+.4f}（λ={lA:g}）")
print("   " + ("⇒ 明显上升，数据量此前受限" if rA > r4 + 0.05 else
               "⇒ **持平或下降 —— 4 篇已经够，加数据买不到更好的拟合**"))

print("\n② 跨篇两两 Spearman（静态表成立的前提）")
for lab, R in (("4 篇", A), ("10 篇", ALL)):
    docs = sorted({x["doc"] for x in R})
    us = [_fit([x for x in R if x["doc"] == d])[2] for d in docs]
    ss = [spearmanr(us[i], us[j]).statistic
          for i in range(len(us)) for j in range(i + 1, len(us))]
    npos = sum(1 for x in ss if x > 0)
    # 符号检验：为正的对数是否显著多于一半
    from scipy.stats import binomtest
    p = binomtest(npos, len(ss), 0.5, alternative="greater").pvalue
    print(f"   {lab}：{len(docs)} 篇 {len(ss)} 对，均值 {np.mean(ss):+.4f} "
          f"中位 {np.median(ss):+.4f}，为正 {npos}/{len(ss)}，符号检验 p={p:.4f}")

s = spearmanr(u4, uA).statistic
print(f"\n③ Spearman(u_4篇, u_10篇) = {s:+.4f}")
print("   " + ("⇒ **方向稳定**，+29.40 站在可复现的方向上" if s > 0.6 else
               "⇒ **表随标定集漂移** —— +29.40 的可复现性存疑，这是重要的负结果"))
