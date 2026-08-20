#!/usr/bin/env python3
"""头内单调性的**网格级证书** —— 比 `armdiag` 的抽样 Kendall τ 强得多。

`scratch_probe_armdiag.py` 逐 (层,头) 抽 **2 万对** token 检查符号一致，得到
τ = 1.000000、翻转 0。但每头约 15 万 token ⇒ 全对约 1.1e10，抽 2 万只能说
"翻转率 < 约 5e-5"，**不能说函数单调**。这个区别是实打实的：一个在训练支撑集上
恰好没翻转的网络，仍可能在别处非单调。

对 `scalar` 族可以做得更硬，因为**头内固定 chunk 时它退化成一元函数**：
`mg = A·z + B`、`rs = log A`、`e` 都是该 (chunk, 层, 头) 的常数，于是

    s' = s + α·σ_h·tanh(φ(z)),      φ(z) = MLP([z, A·z+B, log A, e])
    ds'/ds = 1 + α·sech²(φ(z))·φ'(z)                     （σ_h 约掉了）

所以**只要 `min_z [1 + α·sech²(φ)·φ'(z)] > 0` 就单调**，而 `φ'(z)` 用 autograd
在稠密网格上逐点可算。报告的是：

    margin(z) = 1 + α·sech²(φ(z))·φ'(z)      越接近 0 越危险，≤0 即该点非单调

覆盖范围写清楚：z 网格取真实 trace 里观测到的 z 范围再外扩，(A,B) 取真实
trace 里每个 (chunk, 层, 头) 实测到的值。**这是"在实测状态与网格上的证书"，
不是对全体实数的形式证明** —— 但它比抽样对强一个量级，且能指出最危险的位置。

`kv` 臂不适用：它的 φ 依赖 (k,v) 而非 z，本来就会重排序（armdiag 实测 τ 最低 0.153）。
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


def load(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    arch = sd.get("arch")
    assert arch in CalibScorer.SCALAR_FEATS, \
        f"只对标量族有效（头内退化成一元函数）；arch={arch} 不在 {list(CalibScorer.SCALAR_FEATS)}"
    m = CalibScorer(sd.get("d_kv", 128), sd["L"], sd["H"],
                    n_slots=sd.get("slots", 8), d_m=sd.get("dim", 128),
                    mode="memoryless", arch=arch,
                    scale=sd.get("scale", "head"),
                    alpha_max=sd.get("alpha_max", 1.0))
    m.load_state_dict(sd["state"])
    return m.eval(), arch, sd.get("scale", "head")


def states(traces, n_doc):
    """从 trace 里取真实的 (A_h, B_h, z 范围)，逐 (chunk, 层, 头)。"""
    out, zlo, zhi = [], np.inf, -np.inf
    for f in sorted(glob.glob(os.path.join(ROOT, traces, "doc*.pt")))[:n_doc]:
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            g, t = float(ch["gsig"]), float(ch["thres"])
            for l, pl in enumerate(ch["layers"]):
                s0 = pl["s0"].float(); mu = pl["mu_h"].float(); sg = pl["sig_h"].float()
                for h in range(s0.shape[0]):
                    z = (s0[h] - mu[h]) / sg[h].clamp_min(1e-6)
                    zlo = min(zlo, float(z.min())); zhi = max(zhi, float(z.max()))
                    out.append((l, h, float(sg[h] / g), float((mu[h] - t) / g)))
    return out, zlo, zhi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--traces", default="scratch_ctrl_traces_v2_10")
    ap.add_argument("--n_doc", type=int, default=2, help="取几篇 trace 的实测状态")
    ap.add_argument("--n_grid", type=int, default=4001)
    ap.add_argument("--pad", type=float, default=2.0, help="z 网格在观测范围外再扩多少")
    a = ap.parse_args()

    m, arch, scale = load(a.ckpt)
    alpha = float(m.alpha)
    st, zlo, zhi = states(a.traces, a.n_doc)
    lo, hi = zlo - a.pad, zhi + a.pad
    z = torch.linspace(lo, hi, a.n_grid, dtype=torch.float32)
    print(f"{os.path.basename(os.path.dirname(a.ckpt))}  arch={arch}  "
          f"**scale={scale}**  α={alpha:.4f}")
    print(f"  实测 z ∈ [{zlo:.2f}, {zhi:.2f}]，网格取 [{lo:.2f}, {hi:.2f}] × {a.n_grid} 点")
    print(f"  实测状态 (chunk,层,头) 共 {len(st)} 组；A_h ∈ "
          f"[{min(s[2] for s in st):.4f}, {max(s[2] for s in st):.4f}]  B_h ∈ "
          f"[{min(s[3] for s in st):+.3f}, {max(s[3] for s in st):+.3f}]")

    feats = CalibScorer.SCALAR_FEATS[arch]
    use_emb = arch not in CalibScorer.NO_EMB
    worst, worst_at, n_bad = np.inf, None, 0
    for (l, h, A, B) in st:
        zz = z.clone().requires_grad_(True)
        col = {"z": zz, "mg": A * zz + B,
               "rs": torch.full_like(zz, float(np.log(max(A, 1e-12))))}
        parts = [col[k][:, None] for k in feats]
        if use_emb:
            parts.append(m.emb[l, h][None, :].expand(len(zz), -1))
        phi = m.head(torch.cat(parts, -1)).squeeze(-1)
        dphi, = torch.autograd.grad(phi.sum(), zz)
        # ds'/ds 的推导 —— **两个 scale 的系数不同，不能共用一条公式**：
        #   Δs = α·σ·tanh(φ(z))，z = (s−μ_h)/σ_h ⇒ dφ/ds = φ'(z)/σ_h
        #   ⇒ ds'/ds = 1 + α·σ·sech²(φ)·φ'(z)/σ_h
        # scale="head"   ：σ = σ_h ⇒ **σ_h 恰好约掉**，系数为 1（原公式）。
        # scale="global" ：σ = σ_g ⇒ 系数是 **σ_g/σ_h = 1/A_h**，
        #   而实测 A_h ∈ [0.0026, 1.61] ⇒ 放大最高 385×。
        #   **若这里沿用旧公式，会对 global ckpt 给出错误的「安全」结论。**
        coef = 1.0 if scale == "head" else 1.0 / max(A, 1e-12)
        marg = 1.0 + coef * alpha * (1.0 - torch.tanh(phi) ** 2) * dphi
        mn = float(marg.min())
        if mn <= 0: n_bad += 1
        if mn < worst:
            worst, worst_at = mn, (l, h, A, B, float(z[int(marg.argmin())]))
    l, h, A, B, zs = worst_at
    print(f"\n  min over 全部状态 × 全部网格点  ds'/ds = {worst:+.6f}")
    print(f"    最危险处：层 {l} 头 {h}  A={A:.4f} B={B:+.3f}  z={zs:+.3f}")
    print(f"    非单调的状态数 {n_bad}/{len(st)}")
    if worst > 0:
        print(f"\n  ⇒ **在实测 (A,B) 与该 z 网格上，头内严格单调**（余量 {worst:.4f}）。"
              f"\n     这是网格级证书，不是对全体实数的形式证明；覆盖范围如上。")
    else:
        print(f"\n  ⇒ 存在非单调点 ⇒ 「零重排」只是抽样没抽到，不能当作构造性保证。")


if __name__ == "__main__":
    raise SystemExit(main())
