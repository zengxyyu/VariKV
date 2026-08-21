"""把任务教师解出的逐头边际效用 `u` 变成可直接注入评测的配额增量表（2026-08-21）。

为什么第一枪不训练网络：`VARIKV_QUOTA_INJECT` 吃的就是一张 `[L*H]` 增量表，
`project_quota` 负责保预算。而 2026-08-18 已经测到「组内整个方法坍缩成 112 个
整数」—— 静态逐头表与 637,828 参数的网络逐样本配对 **+0.00 [−2.00,+2.00]**。
所以静态表既是最快的下游读数，也几乎不损失上限：**若 u 是有用的标签，静态
u 表就该赢；若静态 u 表都不赢，训网络也救不回来。**

用法：
    python scratch_u_to_table.py --json scratch_adv_grad_bulk.json \\
        --alloc l2 --mb 0.01 --out varikv/tab_u_l2_mb01.npy
"""
import argparse, json, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def fit_u(recs, lam=None, folds=5, seed=0):
    """岭回归反解 `u`，λ 由留出方向上的 R² 选。返回 (u, R², λ)。

    设计矩阵各行和为 0 ⇒ `1 ∈ null(X)` ⇒ 岭解落在行空间、天然 `⊥ 1`，
    所以规范（`u ∼ u + κ1`）是自动固定的；再显式中心化只是写明这一点。
    """
    X = np.array([r["d"] for r in recs], dtype=np.float64)
    y = np.array([r["A"] for r in recs], dtype=np.float64)
    X = X - X.mean(0)
    sx = X.std() or 1.0
    Xn = X / sx

    def cv(l_):
        rs = np.random.default_rng(seed); idx = rs.permutation(len(y))
        pred = np.zeros_like(y)
        for f in range(folds):
            te = idx[f::folds]; tr = np.setdiff1d(idx, te)
            Xt, yt = Xn[tr], y[tr] - y[tr].mean()
            w = np.linalg.solve(Xt.T @ Xt + l_ * np.eye(Xn.shape[1]), Xt.T @ yt)
            pred[te] = Xn[te] @ w + y[tr].mean()
        return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    # **网格延到 1e5**（2026-08-21）：原来只到 100，而 psyn 教师的最优 λ 是
    # 1000，于是既选错了 λ，又让「λ 顶到上限」被误读成「没信号」。
    lams = [lam] if lam else [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
    r2s = [cv(l_) for l_ in lams]
    i = int(np.argmax(r2s)); lam = lams[i]
    u = np.linalg.solve(Xn.T @ Xn + lam * np.eye(Xn.shape[1]),
                        Xn.T @ (y - y.mean())) / sx
    return u - u.mean(), r2s[i], lam


def build_delta(u, mb_frac, B, alloc, topk=8):
    """由 `u` 造保预算的配额增量 `Δb`，`½‖Δb‖₁ = mb_frac·B`，`Σ Δb = 0`。

    `l2`   —— `Δb ∝ Π u`：L2 信赖域下 `max uᵀΔb` 的解，稠密、保守。
    `topk` —— `+M/k` 给 u 最大的 k 个头，`−M/k` 给最小的 k 个：**L1** 信赖域下的
              精确解（线性目标在 L1 球上的最优点是角点）。更激进，也更容易越界。

    两个都给，是因为「哪个信赖域才是对的」由损伤实际跟着哪个量走决定 —— 已测到
    损伤跟 `‖Δb‖₁` 走（λ 缩幅度无效而 `TRUST_MB` 有效），这偏向 `topk`；但角点解
    把全部搬动压在少数头上，与「修正必须全头一致」的 P0 结论相抵。所以实测。
    """
    M = mb_frac * B
    if alloc == "l2":
        d = u / (np.abs(u).sum() / 2 + 1e-30) * M
    elif alloc == "topk":
        d = np.zeros_like(u)
        o = np.argsort(u)
        d[o[-topk:]] = M / topk
        d[o[:topk]] = -M / topk
    else:
        raise ValueError(alloc)
    d = d - d.mean()                     # 严格 Σ=0（浮点残差归零）
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="scratch_adv_teacher.py --mode grad 的输出")
    ap.add_argument("--eps", type=float, default=None, help="只用这一档扰动尺度的方向")
    ap.add_argument("--alloc", default="l2", choices=["l2", "topk"])
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--mb", type=float, default=0.01, help="½‖Δb‖₁ / B")
    ap.add_argument("--B", type=float, default=None,
                    help="基线保留总数。不给则从记录里的搬动量反推")
    ap.add_argument("--per_doc", action="store_true",
                    help="逐篇各解一个 u 再平均（默认把所有篇的方向合在一起解）。"
                         "合解假设 u 跨文档共享；逐篇平均则允许文档间差异，"
                         "并顺带给出跨文档一致性。")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = json.load(open(os.path.join(ROOT, a.json)))
    recs = [r for r in recs if r.get("mode") == "grad"]
    if a.eps is not None:
        recs = [r for r in recs if abs(r.get("eps", -1) - a.eps) < 1e-12]
    assert recs, "没有匹配的记录"
    docs = sorted({r["doc"] for r in recs})
    print(f"{len(recs)} 个方向，{len(docs)} 篇，G={len(recs[0]['d'])}")

    if a.per_doc:
        us, r2s = [], []
        for d_ in docs:
            sub = [r for r in recs if r["doc"] == d_]
            if len(sub) < 20:
                print(f"  doc{d_} 只有 {len(sub)} 个方向，跳过"); continue
            u_, r2_, lam_ = fit_u(sub)
            us.append(u_); r2s.append(r2_)
            print(f"  doc{d_}: n={len(sub)} CV R² {r2_:+.4f} (λ={lam_:g})")
        U = np.stack(us)
        from scipy import stats as st
        if len(us) > 1:
            cc = [st.spearmanr(U[i], U[j])[0]
                  for i in range(len(us)) for j in range(i + 1, len(us))]
            print(f"  **跨文档 u 的两两 Spearman：中位 {np.median(cc):+.3f} "
                  f"[{min(cc):+.3f}, {max(cc):+.3f}]**")
            print(f"  ⇒ 这直接决定静态表可不可行：接近 0 就说明 u 依赖文档、"
                  f"静态表不成立，必须回到条件化控制器。")
        u = U.mean(0); u -= u.mean(); r2 = float(np.mean(r2s))
    else:
        u, r2, lam = fit_u(recs)
        print(f"  合解：CV R² {r2:+.4f} (λ={lam:g})")

    B = a.B if a.B else float(np.median([r["mb"] for r in recs])) / \
        float(np.median([r.get("eps", 0.01) for r in recs]))
    d = build_delta(u, a.mb, B, a.alloc, a.topk)
    print(f"  B≈{B:.0f}  Δb: ½‖·‖₁={np.abs(d).sum()/2:.0f} "
          f"Σ={d.sum():+.3e} 极差 [{d.min():+.1f}, {d.max():+.1f}] "
          f"实际搬动的头 {(np.abs(d)>0.5).sum()}/{len(d)}" f"（判据 |db_h|>0.5 条；**不是字面非零** —— 保预算的中心化让每项都非零，字面计数恒为 112、毫无信息）")
    np.save(os.path.join(ROOT, a.out), d.astype(np.float32))
    print(f"写出 {a.out}（注入用 VARIKV_QUOTA_INJECT，project_quota 保预算）")


if __name__ == "__main__":
    main()
