#!/usr/bin/env python3
"""三张配额表的**逐层结构**对比 —— 查「ρ=0.5 表为什么主动有害」。

**为什么不是查方向相关**：已有数据已经否掉那条 —— Spearman(kv 表, psyn 表)
= −0.0454、Spearman(kv 表, ρ=0.5 表) = −0.0291，**两张非获胜表都与获胜方向
正交**，所以「与好方向反向」区分不了「无害」与「有害」。

**这里查的是分布结构**：112 = 28 层 × 4 头。已知先验（P0）信号集中在晚层
（24/26/27）。若 ρ=0.5 表系统性地**从晚层抽走配额**而 psyn 不，那就是机制。

**判据先写死（三条，各配阴性对照）**：
  ① **层集中度**：Σ_layer |Σ_h Δb| / Σ|Δb|。对照 = 把 112 个 Δb **随机置换**
     1000 次的分布；只有超出置换 95 分位才算「有结构」。
  ② **晚层净流向**：晚层（≥24）的 Σ Δb 符号与大小，同样对置换分位。
  ③ **三表可分性**：若三张表在 ①② 上都落在置换分布内 ⇒ **记为未解，不硬凑**。
"""
import numpy as np

TAB = {"kv": "varikv/tab_u_l2.npy",
       "psyn": "varikv/tab_u_psyn_l2.npy",
       "r05": "varikv/tab_u_r05_l2.npy"}
L, H = 28, 4
LATE = 24          # 晚层起点（P0 的 24/26/27）
RNG = np.random.default_rng(0)
NPERM = 1000


def stats(d):
    """→ (层集中度, 晚层净流, 晚层占绝对量比例)。d 形状 [112]，按 (层,头) 展开。"""
    m = d.reshape(L, H)
    per_layer = m.sum(1)                       # 每层净流
    conc = np.abs(per_layer).sum() / (np.abs(d).sum() + 1e-30)
    late_net = per_layer[LATE:].sum()
    late_abs = np.abs(m[LATE:]).sum() / (np.abs(d).sum() + 1e-30)
    return conc, late_net, late_abs


def pct(val, null):
    """val 在置换零分布里的分位（0–1）。"""
    return float((null < val).mean())


print(f"逐层结构（L={L} H={H}，晚层 = 层 ≥{LATE}；置换对照 {NPERM} 次）")
print(f"{'表':>6s}{'层集中度':>10s}{'分位':>7s}{'晚层净流':>11s}{'分位':>7s}"
      f"{'晚层|Δb|占比':>13s}{'分位':>7s}")
out, NUL = {}, {}
for nm, f in TAB.items():
    d = np.load(f).ravel().astype(float)
    assert d.shape == (L * H,), d.shape
    c, ln, la = stats(d)
    # **零分布必须逐表各算一份** —— 它是「把这张表自己的 112 个值随机置换」，
    # 依赖该表的取值分布。首版把 `nulls` 当循环变量，判词块用的是**最后一张表**
    # 的零分布去判所有三张，漏判了 psyn 的两个显著项。
    nulls = np.array([stats(RNG.permutation(d)) for _ in range(NPERM)])
    NUL[nm] = nulls
    out[nm] = (c, ln, la)
    print(f"{nm:>6s}{c:>10.4f}{pct(c, nulls[:, 0]):>7.3f}"
          f"{ln:>+11.1f}{pct(ln, nulls[:, 1]):>7.3f}"
          f"{la:>13.4f}{pct(la, nulls[:, 2]):>7.3f}")
print(f"\n（随机置换下层集中度期望 ≈ {np.mean(nulls[:, 0]):.4f}，"
      f"晚层|Δb|占比期望 ≈ {(L - LATE) / L:.4f}）")

print("\n判词（由数字生成）：")
NAMES = ("层集中度", "晚层净流", "晚层|Δb|占比")
hit = {}
for nm, v in out.items():
    h = [NAMES[i] for i in range(3)
         if pct(v[i], NUL[nm][:, i]) > 0.95 or pct(v[i], NUL[nm][:, i]) < 0.05]
    hit[nm] = h
    print(f"  {nm:>5s}: " + ("超出置换对照的项 = " + "、".join(h) if h else "三项全落在置换分布内"))
print()
if not hit["r05"]:
    print("  ⇒ **有害的那张（r05）在这三个逐层量上毫无异常** ——")
    print("     按先写死的第③条：**「ρ=0.5 表为什么主动有害」记为未解，不硬凑。**")
if hit["psyn"]:
    print(f"  ⇒ 但无害的那张（psyn）有结构：{'、'.join(hit['psyn'])}。")
    print("     这是**关于 psyn 为什么安全**的线索，不是关于 r05 为什么有害的答案，")
    print("     **两者不要混为一谈**。")
