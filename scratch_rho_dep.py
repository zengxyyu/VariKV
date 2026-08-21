"""u 是否随预算 ρ 变化？—— 匹配对：`labc_d0`(ρ=0.1) vs `labrho_0.5`(ρ=0.5)。

两个文件同为 chunk 模式、5 篇、2400 条，**只有 ρ 不同** ⇒ 满足「一次只变一个变量」。
判据先写死：
  · 若 Spearman(u_0.1, u_0.5) 高（且高于篇内分半的噪声地板）⇒ u 与预算无关，
    一张静态表用在全 ratio 上是**对的**，高 ratio 的伤另有原因。
  · 若低 ⇒ 表在 ρ=0.1 标定却用在 ρ=0.3~0.75 上是**规格错误**，
    可行动项是**逐 ratio 各标一张表**。
噪声地板由**同一个 ρ 内**按篇分半重拟给出 —— 没有它，任何相关系数都不可读。
"""
import json
import sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/home/ubuntu/zxy/vlm-memory")
from scratch_u_to_table import fit_u                      # noqa: E402

F = {0.1: "scratch_labc_d0.json", 0.5: "scratch_labrho_0.5.json"}
R, U = {}, {}
for r, f in F.items():
    R[r] = json.load(open(f))
    docs = sorted({x["doc"] for x in R[r]})
    u, r2, lam = fit_u(R[r])
    U[r] = u
    print(f"ρ={r}  n={len(R[r])}  docs={docs}  R²={r2:+.4f}  λ={lam}  "
          f"|u| 中位={np.median(np.abs(u)):.5f}  ‖u‖₂={np.linalg.norm(u):.4f}")

print()
# ── 噪声地板：同一 ρ 内按篇分半 ──────────────────────────────────
print("噪声地板（同一 ρ 内按篇分半重拟，篇不重叠）")
floor = {}
for r in F:
    docs = sorted({x["doc"] for x in R[r]})
    h = len(docs) // 2
    a = [x for x in R[r] if x["doc"] in docs[:h]]
    b = [x for x in R[r] if x["doc"] in docs[h:]]
    ua, _, _ = fit_u(a)
    ub, _, _ = fit_u(b)
    s = spearmanr(ua, ub).statistic
    p = np.corrcoef(ua, ub)[0, 1]
    floor[r] = s
    print(f"  ρ={r}  篇{docs[:h]} vs {docs[h:]}   Spearman={s:+.4f}  Pearson={p:+.4f}")

print()
s = spearmanr(U[0.1], U[0.5]).statistic
p = np.corrcoef(U[0.1], U[0.5])[0, 1]
print(f"跨 ρ：Spearman(u_0.1, u_0.5) = {s:+.4f}   Pearson = {p:+.4f}")
fl = np.mean(list(floor.values()))
print(f"噪声地板均值 = {fl:+.4f}")
# 去衰减：真相关 ≈ 观测 / sqrt(信度积)，信度用分半相关的 Spearman-Brown 校正
rel = {r: (2 * v / (1 + v)) if v > -1 else 0.0 for r, v in floor.items()}
den = (max(rel[0.1], 1e-9) * max(rel[0.5], 1e-9)) ** 0.5
print(f"信度（Spearman-Brown）ρ=0.1 {rel[0.1]:+.4f} / ρ=0.5 {rel[0.5]:+.4f}"
      f"  ⇒ 去衰减后 {s / den if den > 0 else float('nan'):+.4f}")
print()
if fl <= 0.05:
    print("判词：**噪声地板本身就 ≈0 ⇒ 这个仪器分辨不了跨 ρ 的问题**，")
    print("      跨 ρ 相关低不能读成「u 随预算变」—— 它同一个 ρ 内也复现不了。")
elif s >= 0.6 * fl:
    print("判词：跨 ρ 相关达到同 ρ 地板的六成以上 ⇒ **u 主要与预算无关**，")
    print("      一张表用在全 ratio 是合规格的，高 ratio 的伤另有原因。")
else:
    print("判词：跨 ρ 相关**显著低于**同 ρ 地板 ⇒ **u 随预算变**，")
    print("      ρ=0.1 标定的表用在高 ratio 上是规格错误 ⇒ 逐 ratio 各标一张。")
