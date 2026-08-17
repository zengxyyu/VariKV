#!/usr/bin/env python3
"""`scalar` 与 `kv` 两条通路：是同一个机制，还是殊途同归？

两臂在 Retr.KV @0.1 上分别是 +4.73 ± 0.41 与 +4.47 ± 0.09，配对差 −0.27 ± 0.34
（不可分）。但**总分相同不等于同一个函数** —— 极端反例：一臂改对前 50 题、另一臂
改对后 50 题，平均分一样而信息完全互补。

逐样本已经测过（`scbench_kv` s0，n=100）：Pearson +0.723，逐样本 Δ 完全相同
79/100，但"被改好"的样本 Jaccard 只有 0.500。**中间态**，所以要往下一层看。

这里看**驱逐掩码**本身，比任务分数细得多（分数是 5 题一格的粗粒度，掩码是逐条 KV）：

    S_base   = {s⁰ > τ}
    S_arm    = {s⁰ + Δs_arm > τ}
    F_arm    = S_arm △ S_base            该臂**翻动**的条目集合

关键统计量是 **Jaccard(F_scalar, F_kv)** —— 两臂翻的是不是同一批条目。
`J≈1` ⇒ 同一机制；`J≈0` 而任务分相近 ⇒ 两条独立通路碰巧同分。

**一个必须声明的近似**：真实流水线是在**修正后的分数**上重算全局阈值，而这里用
trace 里存的 `τ`（在 s⁰ 上算的）。两臂用同一个 τ、口径一致，所以比较是公平的；
但绝对翻转率会与真实值略有出入。teacher trace 只存了每 (chunk,层,头) 768 个候选，
无法重算全局阈值，这是数据本身的限制。

另外报 `Δs` 的直接相关 —— 连续量，比二值翻转敏感得多。
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


def deltas(m, arch, l, k, v, s0, mg, st):
    x = m.feat(m.raw(k, v)) if arch in ("kv", "k", "v") else None
    r = m.read(m.init_state(l), None)
    with torch.no_grad():
        return m.delta(x, r, s0, margin=mg, stats=st)


def jac(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["scalar", "kv"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traces", default="scratch_ctrl_traces_v2_10")
    ap.add_argument("--n_doc", type=int, default=3)
    ap.add_argument("--near_only", action="store_true", default=True,
                    help="只用近阈值候选（后半段随机候选几乎不可能翻）")
    a = ap.parse_args()

    ms = {}
    for A in a.arms:
        p = os.path.join(ROOT, f"varikv/d10_{A}_s{a.seed}.pt/memoryless.pt")
        assert os.path.exists(p), f"缺 {p}"
        ms[A] = load(p)
        print(f"  {A:<7} arch={ms[A][1]}  参数 {ms[A][0].n_params():,}")

    D = {A: [] for A in a.arms}
    S0, TH = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, a.traces, "doc*.pt")))[:a.n_doc]:
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            g, t = float(ch["gsig"]), float(ch["thres"])
            for l, pl in enumerate(ch["layers"]):
                n = pl["n_near"] if a.near_only else pl["s0"].shape[1]
                k = pl["k"][:, :n].float(); v = pl["v"][:, :n].float()
                s0 = pl["s0"][:, :n].float()
                st = (pl["mu_h"].float(), pl["sig_h"].float(), torch.tensor(g))
                mg = (s0 - t) / g
                for A, (m, arch) in ms.items():
                    D[A].append(deltas(m, arch, l, k, v, s0, mg, st).reshape(-1))
                S0.append(s0.reshape(-1)); TH.append(torch.full((s0.numel(),), t))
    s0 = torch.cat(S0); th = torch.cat(TH)
    Dv = {A: torch.cat(D[A]) for A in a.arms}
    n = len(s0)
    print(f"\n候选 {n:,} 条（{a.n_doc} 篇 × 近阈值子集）")

    # **等预算 Top-B，不用固定 τ。** 真实流水线在 `s'=s⁰+Δs` 上**重算**阈值，所以
    # `Δs` 的全局常数偏移是**规范自由度**——对真实决策的影响恰好为零。用固定 τ 判定
    # 会把这个不可辨识的偏移当成真实效应：实测 Δs 均值 −0.0665（= −0.76×标准差），
    # 而 τ 之上的候选离 τ 中位只有 0.0089，于是"几乎全被压下去"，得到 63:91,944 的
    # 荒谬不对称。真实 Top-B 下 |S'|=|S|=B，翻上必须与翻下数量相当。
    # 这里对每条臂取 s' 的全局 top-|base|，预算与基线严格相等。
    base = s0 > th
    B = int(base.sum())
    def topB(x):
        m_ = torch.zeros_like(base)
        m_[torch.topk(x, B).indices] = True
        return m_
    base = topB(s0)          # 与各臂同口径：基线也用 top-B 而不是 s0>τ
    F = {}
    print(f"\n{'臂':<8}{'Δs 标准差':>12}{'翻转率':>10}{'翻上':>8}{'翻下':>8}")
    SEL = {}
    for A in a.arms:
        sel = topB(s0 + Dv[A])
        SEL[A] = sel
        F[A] = sel ^ base
        print(f"{A:<8}{float(Dv[A].std()):>12.4f}{float(F[A].float().mean())*100:>9.3f}%"
              f"{int((sel & ~base).sum()):>8}{int((~sel & base).sum()):>8}")

    A1, A2 = a.arms[0], a.arms[1]
    print(f"\n【Δs 连续量】")
    d1, d2 = Dv[A1].numpy(), Dv[A2].numpy()
    print(f"  Pearson({A1}, {A2}) = {np.corrcoef(d1, d2)[0,1]:+.4f}")
    print(f"  余弦相似度            = "
          f"{float(np.dot(d1,d2)/(np.linalg.norm(d1)*np.linalg.norm(d2))):+.4f}")
    print(f"\n【翻转集合】—— 两臂翻的是不是同一批条目")
    print(f"  |F_{A1}| = {int(F[A1].sum()):,}   |F_{A2}| = {int(F[A2].sum()):,}"
          f"   交 {int((F[A1]&F[A2]).sum()):,}")
    print(f"  **Jaccard(F_{A1}, F_{A2}) = {jac(F[A1], F[A2]):.4f}**")
    # 随机基线：两个同样大小、独立随机的翻转集合会有多大 Jaccard
    p1 = float(F[A1].float().mean()); p2 = float(F[A2].float().mean())
    rnd = p1 * p2 / (p1 + p2 - p1 * p2)
    print(f"  独立随机基线（同样大小）= {rnd:.4f}"
          f"   ⇒ 实测是它的 {jac(F[A1],F[A2])/max(rnd,1e-9):.1f} 倍")
    print(f"\n  同向翻转（都翻上或都翻下）占交集的 "
          f"{float(((SEL[A1]==SEL[A2]) & F[A1] & F[A2]).float().sum() / max(int((F[A1]&F[A2]).sum()),1)):.3f}")
    print(f"\n【保留集本身】J(S_{A1}, S_{A2}) = {jac(SEL[A1], SEL[A2]):.4f}"
          f"   （|S| = {B:,}，三者严格等预算）")
    print(f"  Δs 去掉全局均值后的相关（规范无关部分）: "
          f"{np.corrcoef((Dv[A1]-Dv[A1].mean()).numpy(), (Dv[A2]-Dv[A2].mean()).numpy())[0,1]:+.4f}")
    print("\n判读：Jaccard 接近 1 ⇒ 同一机制；接近随机基线 ⇒ 两条独立通路碰巧同分；"
          "\n      中间 ⇒ 部分共享。注意 Δs 相关高但翻转 Jaccard 低是可能的——"
          "\n      多数条目离阈值远，改多少都不翻，决定权在边界那一小撮。")


if __name__ == "__main__":
    raise SystemExit(main())
