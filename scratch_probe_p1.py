#!/usr/bin/env python3
"""P1 三个零 GPU 探针 —— 各钉死 `ICLR_PLAN.md` 里的一条命题。

**P1-1  修正权限是否正比 σ_h。** `Δs = α·σ_h·tanh(·)` ⇒ 逐头上界 `α·σ_h`，实测跨
112 个 (层,头) 差 **413×**（0.00066–0.27136，同一 chunk 内）。所以"有界保守修正"
不是全局性质：`α·σ_h = 1.18·σ_g` 的头能被推过一个全局标准差，`0.003·σ_g` 的头
动不了任何东西。**可检验推论：翻转应高度集中在高 σ_h 的头上。**

**P1-2  集合层面的 aliasing（逐 token 那次测的是错靶子）。** 早先测
`Var(U | z,层,头,A,B)` 相对 `Var(U | z,层,头)`，超出置换对照只降 2.8% ⇒ `(A,B)` 对
**逐 token 效用**几乎没有增量信息。但若机制在预算层面，该测的是**逐头应得名额**：

    B*_h = 按教师 U 做全局 top-B 后，头 h 拿到几个
    B0_h = 按 s⁰ 做同样的 top-B 后，头 h 拿到几个
    ΔB_h = B*_h − B0_h                     ← 教师想要的**头级重分配**

问 `(A_h, B_h)` 能否解释 `ΔB_h` 在**头身份之外**的变化。做法是逐 (层,头) 对
`ΔB_h ~ (log A_h, B_h)` 做 OLS，报 R²，**并与置换对照比**（把 (A,B) 在同头各 chunk
之间打乱——加自变量必然抬高 R²，裸 R² 无意义）。用回归而不是分箱，是因为每个
(层,头) 只有几十个 chunk 观测，分箱会稀疏。

**P1-3  修正场的有效秩。** 头内固定 chunk 时网络退化成一元函数
`w(z) = tanh(φ(z; A,B,e))`，所以整个方法就是**一族由 (A,B) 索引的一维形变**。
把它在固定 z 网格上求值排成矩阵 `[状态 × 网格]` 做 SVD：若前几个奇异值吃掉绝大部分
方差，「低维修正流形」就从假说变成测量。**同时报去掉 σ_h 的版本** —— σ_h 是逐状态的
标量倍数，留着它会人为抬高 rank-1 占比。

三个探针共用同一批 trace 与同一个 ckpt，互相口径一致。
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.calib_scorer import CalibScorer                      # noqa: E402


def load(path):
    sd = torch.load(path, map_location="cpu")
    m = CalibScorer(sd.get("d_kv", 128), sd["L"], sd["H"],
                    n_slots=sd.get("slots", 8), d_m=sd.get("dim", 128),
                    mode="memoryless", arch=sd["arch"])
    m.load_state_dict(sd["state"])
    return m.eval(), sd["arch"]


def collect(m, arch, traces, n_doc):
    """逐 (chunk, 层, 头) 收 s0 / U / Δs / A / B，并记录归属。"""
    rows, cid = [], 0
    for f in sorted(glob.glob(os.path.join(ROOT, traces, "doc*.pt")))[:n_doc]:
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            g, t = float(ch["gsig"]), float(ch["thres"])
            for l, pl in enumerate(ch["layers"]):
                n = pl["n_near"]
                k = pl["k"][:, :n].float(); v = pl["v"][:, :n].float()
                s0 = pl["s0"][:, :n].float(); U = pl["U"][:, :n].float()
                mu = pl["mu_h"].float(); sg = pl["sig_h"].float().clamp_min(1e-6)
                st = (mu, sg, torch.tensor(g))
                x = m.feat(m.raw(k, v)) if arch in ("kv", "k", "v") else None
                with torch.no_grad():
                    ds = m.delta(x, m.read(m.init_state(l), None), s0,
                                 margin=(s0 - t) / g, stats=st)
                for h in range(s0.shape[0]):
                    rows.append(dict(cid=cid, l=l, h=h, n=n,
                                     s0=s0[h], U=U[h], ds=ds[h],
                                     A=float(sg[h] / g), B=float((mu[h] - t) / g),
                                     sig_h=float(sg[h]), sig_g=g, tau=t))
            cid += 1
    return rows


# ─────────────────────────────────────────────── P1-1
def p1_flips(rows, alpha):
    s0 = torch.cat([r["s0"] for r in rows])
    ds = torch.cat([r["ds"] for r in rows])
    tau = torch.cat([torch.full_like(r["s0"], r["tau"]) for r in rows])
    hid = np.concatenate([np.full(len(r["s0"]), i) for i, r in enumerate(rows)])
    B = int((s0 > tau).sum())
    top = lambda x: torch.zeros_like(s0, dtype=torch.bool).index_fill_(
        0, torch.topk(x, B).indices, True)
    base, sel = top(s0), top(s0 + ds)
    flip = (base ^ sel).numpy()
    auth = np.array([alpha * r["sig_h"] / r["sig_g"] for r in rows])   # α·σ_h/σ_g
    fr = np.array([flip[hid == i].mean() for i in range(len(rows))])
    fn = np.array([flip[hid == i].sum() for i in range(len(rows))], float)
    q = np.quantile(auth, [0, .2, .4, .6, .8, 1.0])
    print(f"\n【P1-1】修正权限 α·σ_h/σ_g 与翻转率  （{len(rows):,} 个 (chunk,层,头) 状态，"
          f"共 {int(fn.sum()):,} 次翻转）")
    print(f"{'权限五分位':>14}{'α·σ_h/σ_g 中位':>16}{'该组翻转率':>12}{'占全部翻转':>12}")
    for i in range(5):
        mk = (auth >= q[i]) & (auth <= q[i + 1] if i == 4 else auth < q[i + 1])
        print(f"{'Q'+str(i+1):>14}{np.median(auth[mk]):>16.4f}"
              f"{fr[mk].mean()*100:>11.2f}%{fn[mk].sum()/fn.sum()*100:>11.1f}%")
    from scipy.stats import spearmanr
    print(f"  Spearman(权限, 该状态翻转率) = {spearmanr(auth, fr).statistic:+.4f}")
    print(f"  最高 20% 权限的头贡献了 {fn[auth >= q[4]].sum()/fn.sum()*100:.1f}% 的翻转"
          f"（均匀应为 20%）")


# ─────────────────────────────────────────────── P1-2
def p1_budget(rows, n_shuf=20, seed=0):
    """ΔB_h（教师想要的头级重分配）能否被 (A_h,B_h) 解释，在头身份之外。"""
    s0 = torch.cat([r["s0"] for r in rows]); U = torch.cat([r["U"] for r in rows])
    tau = torch.cat([torch.full_like(r["s0"], r["tau"]) for r in rows])
    hid = np.concatenate([np.full(len(r["s0"]), i) for i, r in enumerate(rows)])
    B = int((s0 > tau).sum())
    top = lambda x: torch.zeros_like(s0, dtype=torch.bool).index_fill_(
        0, torch.topk(x, B).indices, True)
    b0, bs = top(s0).numpy(), top(U).numpy()
    dB = np.array([bs[hid == i].sum() - b0[hid == i].sum() for i in range(len(rows))],
                  float)
    key = np.array([r["l"] * 1000 + r["h"] for r in rows])
    X = np.stack([np.log(np.maximum([r["A"] for r in rows], 1e-12)),
                  [r["B"] for r in rows]], 1)
    rng = np.random.default_rng(seed)

    def r2(Xm):
        """逐 (层,头) 做 OLS，合并残差平方和 → 总体 R²（相对逐头均值模型）。"""
        ss_r = ss_t = 0.0
        for k in np.unique(key):
            mk = key == k
            y = dB[mk]
            if len(y) < 5:
                continue
            A_ = np.c_[np.ones(len(y)), Xm[mk]]
            beta, *_ = np.linalg.lstsq(A_, y, rcond=None)
            ss_r += ((y - A_ @ beta) ** 2).sum()
            ss_t += ((y - y.mean()) ** 2).sum()
        return 1 - ss_r / max(ss_t, 1e-12)

    real = r2(X)
    sh = []
    for _ in range(n_shuf):
        Xs = X.copy()
        for k in np.unique(key):           # 只在同 (层,头) 内打乱，保留分布与样本量
            mk = np.flatnonzero(key == k)
            Xs[mk] = X[rng.permutation(mk)]
        sh.append(r2(Xs))
    sh = np.array(sh)
    print(f"\n【P1-2】集合层面：(A_h,B_h) 能否解释教师想要的头级重分配 ΔB_h")
    print(f"  ΔB_h 的分布: 均 {dB.mean():+.2f}  标准差 {dB.std():.2f}  "
          f"|ΔB| 中位 {np.median(np.abs(dB)):.1f}  （每头候选 {rows[0]['n']} 条）")
    print(f"  逐头 OLS  R²(ΔB_h ~ log A_h, B_h) = {real:.4f}")
    print(f"  置换对照                          = {sh.mean():.4f} ± {sh.std():.4f}"
          f"  ({n_shuf} 次)")
    print(f"  **超出对照 = {real - sh.mean():+.4f}**"
          f"   （{(real-sh.mean())/max(sh.std(),1e-9):+.1f} 个对照标准差）")
    print("  对照：早先逐 token 版本（Y = U）超出对照只有 +0.0276")


# ─────────────────────────────────────────────── P1-3
def p1_rank(m, rows, n_grid=201, zlo=-6.0, zhi=6.0):
    """一族由 (A,B) 索引的一维形变，SVD 看有效秩。"""
    feats = CalibScorer.SCALAR_FEATS.get(m.arch)
    if feats is None:
        print("\n【P1-3】跳过：仅对标量族有效（头内才退化成一元函数）"); return
    z = torch.linspace(zlo, zhi, n_grid)
    W, Draw = [], []
    with torch.no_grad():
        for r in rows:
            col = {"z": z, "mg": r["A"] * z + r["B"],
                   "rs": torch.full_like(z, float(np.log(max(r["A"], 1e-12))))}
            parts = [col[k][:, None] for k in feats]
            if m.arch not in CalibScorer.NO_EMB:
                parts.append(m.emb[r["l"], r["h"]][None, :].expand(n_grid, -1))
            w = torch.tanh(m.head(torch.cat(parts, -1)).squeeze(-1))
            W.append(w); Draw.append(float(m.alpha) * r["sig_h"] * w)
    for nm, M_ in (("tanh(φ) 形变本身（去掉 σ_h 尺度）", torch.stack(W)),
                   ("Δs 原样（含 α·σ_h）", torch.stack(Draw))):
        A_ = M_.numpy(); A_ = A_ - A_.mean(0, keepdims=True)   # 去掉共同形变
        s = np.linalg.svd(A_, compute_uv=False)
        ev = s ** 2 / (s ** 2).sum()
        eff = float(np.exp(-(ev * np.log(ev + 1e-30)).sum()))   # 参与比（有效秩）
        print(f"\n【P1-3】{nm}   矩阵 {A_.shape[0]}×{A_.shape[1]}（已减去逐网格均值）")
        print(f"  前 1/2/3/5 个奇异方向解释方差: "
              + " / ".join(f"{ev[:k].sum()*100:.1f}%" for k in (1, 2, 3, 5)))
        print(f"  有效秩（熵参与比 exp(H)）= {eff:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="varikv/d10_scalar_s0.pt/memoryless.pt")
    ap.add_argument("--traces", default="scratch_ctrl_traces_v2_10")
    ap.add_argument("--n_doc", type=int, default=4)
    a = ap.parse_args()
    m, arch = load(os.path.join(ROOT, a.ckpt))
    m.arch = arch
    print(f"{a.ckpt}  arch={arch}  α={float(m.alpha):.4f}")
    rows = collect(m, arch, a.traces, a.n_doc)
    print(f"收集 {len(rows):,} 个 (chunk,层,头) 状态，来自 {a.n_doc} 篇 trace")
    p1_flips(rows, float(m.alpha))
    p1_budget(rows)
    p1_rank(m, rows)


if __name__ == "__main__":
    raise SystemExit(main())
