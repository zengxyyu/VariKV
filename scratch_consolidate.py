#!/usr/bin/env python3
"""把本轮所有要写进文档的数字在**一条代码路径**上重算一遍。

为什么单独写这个：此前的数字散落在十几个一次性片段里，每个片段自带一份
glob 与 bootstrap。写进 `ICLR_PLAN.md` / `CLAUDE.md` 之前必须确认它们
互相一致 —— 一个数字在两处不同就说明至少有一处口径错了。

全部走 `scratch_read_scores.py`（gate 无关、输出文件名无关、匹配为空即抛错）。
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from scratch_read_scores import read_scores, paired          # noqa: E402


def line(lab, o, base, full_mean, base_mean, n_expect=100):
    m, lo, hi, n = paired(o, base)
    star = "*" if (lo > 0 or hi < 0) else " "
    absol = np.mean([o[j] for j in sorted(set(o) & set(base))]) * 100
    hr = full_mean - base_mean
    flag = "" if n >= n_expect else f"  <n={n}!>"
    print(f"  {lab:<30}{m:>+8.2f} [{lo:+.2f},{hi:+.2f}]{star}"
          f"{absol:>8.2f}{m/hr*100:>7.0f}%{n:>5}{flag}")
    return m, lo, hi, n


def panel(ds, ratio, arms, title):
    B = read_scores(ds, "_g8base", ratio)
    F = read_scores(ds, "_g8base", 1.0)
    bm, fm = np.mean(list(B.values())) * 100, np.mean(list(F.values())) * 100
    print(f"\n### {title}   基线 {bm:.2f}  full {fm:.2f}  headroom {fm-bm:+.2f}")
    print(f"  {'臂':<30}{'Δ':>8}{'95% CI':>19}{'绝对':>8}{'恢复':>7}{'n':>5}")
    out = {}
    for lab, tag in arms:
        try:
            out[lab] = (read_scores(ds, tag, ratio),
                        line(lab, read_scores(ds, tag, ratio), B, fm, bm))
        except FileNotFoundError as e:
            print(f"  {lab:<30}  缺：{e}")
    return B, bm, fm, out


print("=" * 78)
B, bm, fm, KV = panel("scbench_kv", 0.2, [
    ("静态表 s0（=完整表 γ=1）", "_p02own"),
    ("静态表 s1", "_p02s1"),
    ("静态表 s2", "_p02s2"),
    ("三种子平均表", "_p02m3"),
    ("网络 scalar s0", "_d10scalar_s0"),
    ("网络 scalar s1", "_d10scalar_s1"),
    ("网络 scalar s2", "_d10scalar_s2"),
    ("网络 v2c s0", "_v2c_s0"),
    ("层内-only（修正版）", "_p02win3"),
    ("层内-only（旧投影）", "_p02win2"),
    ("跨层-only（重构版）", "_p02acr3"),
    ("跨层-only（首版）", "_p02acr2"),
    ("位置索引表 [11,112]", "_p02pos"),
    ("γ=0.5", "_p02x05"),
    ("γ=1.5", "_p02x15"),
    ("γ=2.5", "_p02x25"),
    ("迁移 PrefSuf→KV 预算比", "_p02xferps"),
    ("迁移 PrefSuf→KV 幅度匹配", "_p02xferpsx"),
    ("等幅度分层置换", "_p02perm"),
], "Retr.KV @ρ=0.2")

print("\n  【关键配对】")
for a, b, why in (("静态表 s0（=完整表 γ=1）", "网络 scalar s0", "静态 vs 同源网络 s0"),
                  ("静态表 s1", "网络 scalar s1", "s1"),
                  ("静态表 s2", "网络 scalar s2", "s2"),
                  ("层内-only（修正版）", "静态表 s0（=完整表 γ=1）", "层内 − 完整"),
                  ("跨层-only（重构版）", "静态表 s0（=完整表 γ=1）", "跨层 − 完整"),
                  ("γ=0.5", "静态表 s0（=完整表 γ=1）", "γ0.5 − γ1"),
                  ("γ=1.5", "静态表 s0（=完整表 γ=1）", "γ1.5 − γ1"),
                  ("位置索引表 [11,112]", "静态表 s0（=完整表 γ=1）", "位置 − 扁平"),
                  ("位置索引表 [11,112]", "网络 scalar s0", "位置 − 网络"),
                  ("迁移 PrefSuf→KV 幅度匹配", "静态表 s0（=完整表 γ=1）", "迁移 − 自表")):
    if a in KV and b in KV:
        m, lo, hi, n = paired(KV[a][0], KV[b][0])
        print(f"    {why:<22}{m:>+8.2f} [{lo:+.2f},{hi:+.2f}]"
              f"{'*' if (lo>0 or hi<0) else ' '}  n={n}")

print("\n" + "=" * 78)
_, pbm, pfm, PS = panel("scbench_prefix_suffix", 0.2, [
    ("自表 γ=1", "_ps02tab"),
    ("自表 γ=2.157", "_ps02ownx"),
    ("自表 γ=3.0", "_ps02x3"),
    ("自表 γ=4.5", "_ps02x45"),
    ("迁移 KV→PrefSuf", "_ps02xfer"),
    ("v2c s0", "_v2c_s0"),
], "Retr.PrefSuf @ρ=0.2")
print("\n  【关键配对】")
for a, b, why in (("自表 γ=2.157", "自表 γ=1", "γ2.157 − γ1"),
                  ("自表 γ=3.0", "自表 γ=2.157", "γ3 − γ2.157"),
                  ("自表 γ=4.5", "自表 γ=3.0", "γ4.5 − γ3"),
                  ("迁移 KV→PrefSuf", "自表 γ=2.157", "迁移 − 幅度匹配自表")):
    if a in PS and b in PS:
        m, lo, hi, n = paired(PS[a][0], PS[b][0])
        print(f"    {why:<22}{m:>+8.2f} [{lo:+.2f},{hi:+.2f}]"
              f"{'*' if (lo>0 or hi<0) else ' '}  n={n}")

print("\n" + "=" * 78)
print("\n### 静态表的 ratio 曲线（Retr.KV）")
print(f"  {'ρ':>6}{'基线':>8}{'headroom':>10}{'Δ':>9}{'95% CI':>19}{'恢复':>7}")
for r, tag in ((0.5, "_p05tab"), (0.3, "_p03tab"), (0.2, "_p02own"), (0.1, "_p01tab")):
    Br = read_scores("scbench_kv", "_g8base", r)
    O = read_scores("scbench_kv", tag, r)
    m, lo, hi, n = paired(O, Br)
    b_ = np.mean(list(Br.values())) * 100
    print(f"  {r:>6}{b_:>8.2f}{fm-b_:>+10.2f}{m:>+9.2f}"
          f"{f'[{lo:+.2f},{hi:+.2f}]':>19}{'*' if (lo>0 or hi<0) else ' '}{m/(fm-b_)*100:>6.0f}%")
