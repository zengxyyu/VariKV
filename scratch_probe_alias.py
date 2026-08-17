#!/usr/bin/env python3
"""state aliasing：**全局竞争状态 (A,B) 是否携带局部秩 z 之外的信息？**

这是 `+4.27` 机制假说的直接检验，而且**零 GPU、不训练任何模型**——只用教师 trace。

假说：`affine`/`sz` 这类只看 `(z, 层, 头)` 的打分器有一个信息论上的天花板，因为
同一个 `z` 在不同 chunk 下**面对的全局竞争完全不同**，却被映射成同一个输入状态
（state aliasing）。解除混叠需要

    A_h = σ_h/σ_g        本头分数尺度 ÷ 全局尺度
    B_h = (μ_h−τ)/σ_g    本头中心相对全局淘汰线的位置

注意措辞：`affine` **不是静态方法**——它通过 `z=(s−μ_h)/σ_h` 和输出的 `×σ_h`
已经自适应了运行时的**局部**统计量。它缺的是 `τ` 与 `σ_g`，即**全局**那一半。
所以对立轴是「局部标准化 vs 实际全局竞争状态」，不是「静态 vs 动态」。

度量：教师效用 `U` 的条件方差下降

    R = Var(U | z-bin, 层, 头, A-bin, B-bin) / Var(U | z-bin, 层, 头)

**必须有置换对照。** 多加条件变量会**机械地**降低组内方差（格子变小、样本变少、
样本方差向下偏），所以裸的 R < 1 什么也证明不了。对照做法：把 (A,B) 标签在同一
(层,头) 的各 chunk 之间**随机置换**再算一遍 R_shuf —— 它保留了分箱结构与样本量，
只破坏「(A,B) 与该 chunk 的 U 的对应关系」。真正的信号是 `R` 显著低于 `R_shuf`。
"""
import argparse
import glob
import os

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))


def collect(traces, n_doc, near_only=True):
    """→ 每个近阈值候选一行：(l, h, chunk_id, z, U, A, B)"""
    rows, cid = [], 0
    for f in sorted(glob.glob(os.path.join(ROOT, traces, "doc*.pt")))[:n_doc]:
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            g, t = float(ch["gsig"]), float(ch["thres"])
            for l, pl in enumerate(ch["layers"]):
                s0 = pl["s0"].float(); U = pl["U"].float()
                mu = pl["mu_h"].float(); sg = pl["sig_h"].float().clamp_min(1e-6)
                n = pl["n_near"] if near_only else s0.shape[1]
                for h in range(s0.shape[0]):
                    z = ((s0[h, :n] - mu[h]) / sg[h]).numpy()
                    u = U[h, :n].numpy()
                    A = float(sg[h] / g); B = float((mu[h] - t) / g)
                    rows.append(np.stack([
                        np.full(n, l), np.full(n, h), np.full(n, cid),
                        z, u, np.full(n, A), np.full(n, B)], 1))
            cid += 1
    return np.concatenate(rows, 0).astype(np.float64)


def pooled_var(vals, keys, min_n):
    """按 keys 分组的**合并组内方差**（无偏），只算样本数 ≥ min_n 的组。"""
    order = np.lexsort(keys[::-1])
    k = np.stack(keys, 1)[order]
    v = vals[order]
    bnd = np.r_[0, np.flatnonzero((k[1:] != k[:-1]).any(1)) + 1, len(v)]
    ss = df = 0.0
    for a, b in zip(bnd[:-1], bnd[1:]):
        if b - a < min_n:
            continue
        seg = v[a:b]
        ss += ((seg - seg.mean()) ** 2).sum(); df += (b - a - 1)
    return ss / max(df, 1), df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="scratch_ctrl_traces_v2_10")
    ap.add_argument("--n_doc", type=int, default=4)
    ap.add_argument("--qz", type=int, default=16, help="z 的分位分箱数")
    ap.add_argument("--qab", type=int, default=4, help="A、B 各自的分位分箱数")
    ap.add_argument("--min_n", type=int, default=8)
    ap.add_argument("--n_shuf", type=int, default=5)
    a = ap.parse_args()

    D = collect(a.traces, a.n_doc)
    l, h, cid, z, u, A, B = (D[:, i] for i in range(7))
    print(f"trace {a.traces} 前 {a.n_doc} 篇 → {len(D):,} 个近阈值候选，"
          f"{len(np.unique(cid)):.0f} 个 chunk")
    print(f"  A ∈ [{A.min():.4f}, {A.max():.4f}]（{A.max()/max(A.min(),1e-12):.0f}×）"
          f"　B ∈ [{B.min():+.2f}, {B.max():+.2f}]")

    qb = lambda x, q: np.clip(np.searchsorted(
        np.quantile(x, np.linspace(0, 1, q + 1)[1:-1]), x), 0, q - 1)
    zb, ab, bb = qb(z, a.qz), qb(np.log(np.maximum(A, 1e-12)), a.qab), qb(B, a.qab)

    base, df0 = pooled_var(u, [l, h, zb], a.min_n)
    full, df1 = pooled_var(u, [l, h, zb, ab, bb], a.min_n)
    print(f"\n  Var(U | z,层,头)          = {base:.6f}   (df {df0:,.0f})")
    print(f"  Var(U | z,层,头,A,B)      = {full:.6f}   (df {df1:,.0f})")
    print(f"  裸比值 R = {full/base:.4f}   ← 单看这个没有意义，加条件必然下降")

    # **置换对照**：(A,B) 在同一 (层,头) 的 chunk 之间随机打乱。保留分箱结构与
    # 样本量，只破坏 (A,B) ↔ 该 chunk 的 U 的对应关系。
    rng = np.random.default_rng(0)
    key = l * 1000 + h
    Rs = []
    for _ in range(a.n_shuf):
        ab_s, bb_s = ab.copy(), bb.copy()
        for k in np.unique(key):
            mk = key == k
            cs = np.unique(cid[mk])
            perm = rng.permutation(cs)
            mp = dict(zip(cs, perm))
            src = np.array([mp[c] for c in cid[mk]])
            # 把该 (层,头) 下每个 chunk 的 (A,B) 换成另一个 chunk 的
            look = {c: (ab[mk & (cid == c)][0], bb[mk & (cid == c)][0]) for c in cs}
            ab_s[mk] = [look[s][0] for s in src]
            bb_s[mk] = [look[s][1] for s in src]
        vs, _ = pooled_var(u, [l, h, zb, ab_s, bb_s], a.min_n)
        Rs.append(vs / base)
    Rs = np.array(Rs)
    print(f"  置换对照 R_shuf = {Rs.mean():.4f} ± {Rs.std():.4f}  ({a.n_shuf} 次)")
    excess = Rs.mean() - full / base
    print(f"\n  **超出置换对照的方差下降 = {excess:+.4f}**"
          f"（占置换后残余的 {excess/max(Rs.mean(),1e-9)*100:+.1f}%）")
    print("\n判读：excess ≈ 0 ⇒ (A,B) 不携带 z 之外的信息，state-aliasing 假说不成立；"
          "\n      excess 明显 > 0 ⇒ 同一个局部秩 z 在不同全局竞争状态下确实对应不同效用，"
          "\n      只看 (z,层,头) 的打分器（affine / sz）有信息论上的天花板。")


if __name__ == "__main__":
    raise SystemExit(main())
