#!/usr/bin/env python3
"""四臂报告：层级聚合误差的 context 级配对 bootstrap + 回收率 ρ 的 CI。

**为什么不能直接读 probe 自己打的那两张表。** 两个口径问题：

1. probe 内置的 bootstrap 跑在 `A[:,6]`（**头级**误差）上，而结论要用的是
   **层级聚合**误差（跨头相消后残差流真正看到的量，`LAY[:,2]`）。两者数值接近
   但不是同一个量，不能混着引用。
2. probe 里 `if len(v) != len(base): continue` 会在 scheme 的 context 数与基线
   不一致时**静默跳过**。转导臂第一条样本的 C_q 还没就绪（需要 ≥64 个观测），
   于是 Cq-Lloyd 只有 89 个 context，整条比较被无声丢掉——转导臂因此根本没打出
   C_q 的置信区间。这里改成取 context 交集，并把丢掉的条数打出来。

ρ = (E_eucl − E_Cq)/(E_eucl − E_oracle)，即 C_q 回收了多少「缺 query 信息」造成的
缺口。ρ 的 CI 由同一组 context 重采样下的比值分布给出（ratio of means，不是均值之比
的点估计），所以分母的不确定性也被计入。
"""
import os
import sys

import numpy as np

S = ("position", "random", "eucl-Lloyd", "Cq-Lloyd", "score-oracle")
ARMS = [("kv_trans", "Retr.KV  transductive"), ("kv_held", "Retr.KV  held-out"),
        ("vt_trans", "MultiHop transductive"), ("vt_held", "MultiHop held-out")]
COL = {"层级/损伤": 2, "层级/‖y_full‖": 11}      # LAY: 0=si 1=layer 2=cc 8=ctx 9=nl 11=cc/‖y_full‖
CTX = 8
NBOOT, SEED = 10000, 0


def per_ctx(L, si, col, ctxs):
    """每个 context 先对层取中位 → [n_ctx]，缺该 context 时给 nan。"""
    B = L[L[:, 0] == si]
    out = np.full(len(ctxs), np.nan)
    for j, c in enumerate(ctxs):
        m = B[:, CTX] == c
        if m.any():
            out[j] = np.median(B[m, col])
    return out


def boot(mat, seed=SEED, n=NBOOT):
    """mat: [n_ctx, k] 同一组 context 上的 k 个量 → 返回每列的重采样矩阵 [n,k]。"""
    r = np.random.default_rng(seed)
    idx = r.integers(0, mat.shape[0], (n, mat.shape[0]))
    return mat[idx].mean(1)


def main():
    for name, label in ARMS:
        p = f"scratch_cqv3_{name}_lay.npy"
        if not os.path.exists(p):
            print(f"[skip] {label}: 没有 {p}")
            continue
        L = np.load(p)
        for cname, col in COL.items():
            ctxs = np.unique(L[:, CTX])
            V = np.stack([per_ctx(L, si, col, ctxs) for si in range(len(S))], 1)
            ok = np.isfinite(V).all(1)          # **context 交集**，不是逐 scheme 各取各的
            dropped = int((~ok).sum())
            V = V[ok]
            if len(V) < 10:
                print(f"[skip] {label} {cname}: 有效 context 只有 {len(V)}")
                continue
            B = boot(V)
            print("=" * 96)
            print(f"{label}　|　{cname}　|　n={len(V)} contexts"
                  + (f"　（丢弃 {dropped} 个：某 scheme 在该 context 无数据）" if dropped else ""))
            e, q, o = V[:, 2].mean(), V[:, 3].mean(), V[:, 4].mean()
            print(f"  eucl-Lloyd {e:.4f}   Cq-Lloyd {q:.4f}   score-oracle {o:.4f}")
            for si in (0, 1, 3, 4):
                d = B[:, si] - B[:, 2]
                lo, hi = np.quantile(d, [.025, .975])
                star = "★" if (lo > 0 or hi < 0) else "未分离"
                print(f"  {S[si]:<13} − eucl-Lloyd　{(V[:,si]-V[:,2]).mean():>+8.4f} "
                      f"[{lo:>+7.4f},{hi:>+7.4f}] {star}")
            # 回收率：分母也参与重采样
            rho = (B[:, 2] - B[:, 3]) / np.maximum(B[:, 2] - B[:, 4], 1e-12)
            lo, hi = np.quantile(rho, [.025, .975])
            print(f"  **ρ = (eucl−Cq)/(eucl−oracle) = {(e-q)/(e-o):.1%} "
                  f"[{lo:.1%}, {hi:.1%}]**")


if __name__ == "__main__":
    sys.exit(main())
