"""把逐 chunk 标签变成可注入的配额增量表，并顺带判定序贯一致性（2026-08-21）。

为什么不训网络就能评测：`learned_ctrlcache.py` 的 `VARIKV_QUOTA_INJECT` **本来就
接受 `[C, 112]` 的二维表**（逐 chunk 位置 × 头），每个 chunk 取一行、超出末行则
重复最后一行。而 2026-08-18 已测到「组内整个方法坍缩成 112 个整数」——位置索引表
与 637,828 参数的网络逐样本配对 **+0.00 [−2.00,+2.00]**。所以这条路既最快，
也几乎不损失上限。

顺带回答一个此前一直挂着的未验项：教师标的是**终态一步动作**，而控制器是**逐
chunk 序贯策略**。这里首/中/尾三个 chunk 各解一个 `u`，它们之间的一致性就是答案：

    高度一致 ⇒ 逐 chunk 条件化没必要，一维表足够（也说明终态近似成立）
    差异大   ⇒ 必须用二维表，且教师不能只在终态测

用法：
    python scratch_chunk_table.py --json 'scratch_labc_d*.json' --out varikv/tab_ck.npy
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats as st

ROOT = os.path.dirname(os.path.abspath(__file__))


def fit_u(recs, lam=None, folds=5, seed=0):
    """岭回归反解 `u`，λ 由留出**方向**上的 R² 选。返回 (u, R², λ, n)。

    设计矩阵各行和为 0（保预算）⇒ `1 ∈ null(X)` ⇒ 岭解落在行空间、天然 `⊥ 1`，
    规范 `u ∼ u + κ1` 自动固定；再显式中心化只是把这一点写明。
    CV 折按**方向**分，一个方向的 8 个问句先平均成一个 A —— 把同一方向的问句
    拆到 train/val 两边会泄漏。
    """
    X = np.array([r["d"] for r in recs], dtype=np.float64)
    y = np.array([r["A"] for r in recs], dtype=np.float64)
    X = X - X.mean(0)
    sx = X.std() or 1.0
    Xn = X / sx

    def cv(l_):
        rs = np.random.default_rng(seed)
        idx = rs.permutation(len(y))
        pred = np.zeros_like(y)
        for f in range(folds):
            te = idx[f::folds]
            tr = np.setdiff1d(idx, te)
            Xt, yt = Xn[tr], y[tr] - y[tr].mean()
            w = np.linalg.solve(Xt.T @ Xt + l_ * np.eye(Xn.shape[1]), Xt.T @ yt)
            pred[te] = Xn[te] @ w + y[tr].mean()
        return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    lams = [lam] if lam else [1e-2, 1e-1, 1.0, 10.0, 100.0]
    r2s = [cv(l_) for l_ in lams]
    i = int(np.argmax(r2s))
    lam = lams[i]
    u = np.linalg.solve(Xn.T @ Xn + lam * np.eye(Xn.shape[1]),
                        Xn.T @ (y - y.mean())) / sx
    return u - u.mean(), r2s[i], lam, len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="glob，例如 'scratch_labc_d*.json'")
    ap.add_argument("--rows", type=int, default=11,
                    help="输出表的行数 = 评测时的 chunk 数。scbench_kv 169k/16000≈11，"
                         "prefix_suffix 112k/16000≈8。**行数不对会让后面的 chunk "
                         "重复用最后一行**（注入端是 clamp 不是循环）")
    ap.add_argument("--mb", type=float, default=0.01, help="½‖Δb‖₁ / B（仅定尺度；"
                    "注入端若设 VARIKV_QUOTA_RELMB 会按 chunk 预算重标定，此值只影响相对形状之外的绝对量")
    ap.add_argument("--force_1d", action="store_true", help="强制输出一维表（所有 chunk 同一行）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = []
    for f in sorted(glob.glob(os.path.join(ROOT, a.json))):
        try:
            r = json.load(open(f))
        except Exception as e:
            print(f"  跳过 {os.path.basename(f)}: {e}")
            continue
        r = [x for x in r if x.get("mode") == "chunk"]
        if r:
            recs += r
            print(f"  {os.path.basename(f)}: {len(r)} 条")
    assert recs, "没有 chunk 模式的记录"

    # 逐 (篇, chunk) 解 u；chunk 用**相对位置**归一，因为各篇 chunk 数不同
    U = {}
    for doc in sorted({r["doc"] for r in recs}):
        for ci in sorted({r["chunk"] for r in recs if r["doc"] == doc}):
            sub = [r for r in recs if r["doc"] == doc and r["chunk"] == ci]
            if len(sub) < 40:
                continue
            C = sub[0]["n_chunk"]
            rel = ci / max(C - 1, 1)
            u, r2, lam, n = fit_u(sub)
            U.setdefault(round(rel, 3), []).append((doc, u, r2))
            print(f"    doc{doc} chunk{ci}/{C-1} (rel={rel:.2f}) n={n} "
                  f"CV R²={r2:+.4f} λ={lam:g}")
    rels = sorted(U)
    print(f"\n  相对位置分档: {rels}   每档篇数 {[len(U[r]) for r in rels]}")

    # ---- 序贯一致性：首 / 中 / 尾 的 u 是否一致 ----
    print(f"\n════ 序贯一致性（此前一直挂着的未验项）════")
    bar = {}
    for r_ in rels:
        M = np.stack([u for _, u, _ in U[r_]])
        bar[r_] = M.mean(0)
        print(f"  rel={r_:.2f}: {len(U[r_])} 篇  篇间两两 Spearman 中位 "
              f"{np.median([st.spearmanr(M[i], M[j])[0] for i in range(len(M)) for j in range(i+1, len(M))]) if len(M) > 1 else float('nan'):+.3f}"
              f"   平均 CV R² {np.mean([x[2] for x in U[r_]]):+.3f}")
    if len(rels) > 1:
        print(f"  --- 跨 chunk 位置（同一篇内配对，去掉篇间差异）---")
        for i in range(len(rels)):
            for j in range(i + 1, len(rels)):
                a_, b_ = rels[i], rels[j]
                da = {d: u for d, u, _ in U[a_]}
                db = {d: u for d, u, _ in U[b_]}
                com = sorted(set(da) & set(db))
                if not com:
                    continue
                sp = [st.spearmanr(da[d], db[d])[0] for d in com]
                cs = [float(da[d] @ db[d] / (np.linalg.norm(da[d]) * np.linalg.norm(db[d]) + 1e-30))
                      for d in com]
                print(f"    rel {a_:.2f} vs {b_:.2f}  n={len(com)} 篇  "
                      f"Spearman 中位 {np.median(sp):+.3f}   cos 中位 {np.median(cs):+.3f}")
        print(f"  **判读**：跨 chunk 位置的 Spearman 接近篇间那一档 ⇒ chunk 位置不是"
              f"主要变异源、一维表够用（也说明终态近似成立）；明显更低 ⇒ 必须二维表。")

    # ---- 造表 ----
    if a.force_1d or len(rels) == 1:
        u_all = np.mean([u for r_ in rels for _, u, _ in U[r_]], axis=0)
        u_all -= u_all.mean()
        tab = np.tile(u_all, (a.rows, 1))
        print(f"\n  输出**一维等价**表（每行相同）")
    else:
        # 按相对位置线性插值到 `rows` 行
        xs = np.array(rels)
        Y = np.stack([bar[r_] for r_ in rels])           # [n_rel, 112]
        tgt = np.linspace(0, 1, a.rows)
        tab = np.stack([np.array([np.interp(t, xs, Y[:, h]) for h in range(Y.shape[1])])
                        for t in tgt])
        print(f"\n  输出 [{a.rows}, {Y.shape[1]}] 二维表（相对位置线性插值）")

    # 每行独立归一到目标搬动量并严格 Σ=0
    tab = tab - tab.mean(1, keepdims=True)
    B = 1.0 / a.mb                                        # 只定相对尺度
    for i in range(tab.shape[0]):
        l1 = np.abs(tab[i]).sum() / 2
        if l1 > 0:
            tab[i] = tab[i] / l1 * (a.mb * B)
        tab[i] -= tab[i].mean()
    print(f"  每行 Σ={np.abs(tab.sum(1)).max():.2e}（须≈0）  "
          f"½‖·‖₁ 范围 [{np.abs(tab).sum(1).min()/2:.3f}, {np.abs(tab).sum(1).max()/2:.3f}]")
    np.save(os.path.join(ROOT, a.out), tab.astype(np.float32))
    print(f"写出 {a.out}  形状 {tab.shape}")
    print(f"  注入：VARIKV_QUOTA_INJECT={a.out} 且 **必须设 VARIKV_QUOTA_RELMB**"
          f"（表只带方向，幅度按 chunk 预算重标定）")


if __name__ == "__main__":
    main()
