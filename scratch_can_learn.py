"""控制器可学性判决：`chead` 的两个特征能不能预测逐 chunk 的配额效用 `u`（2026-08-21）。

为什么这是**训练实验最便宜的形式**：`chead` 的输入只有两个逐头标量
（`calib_scorer.py:200-211`）

    rs  = log(σ_h/σ_g) = log A_h        mgm = (μ_h − τ)/σ_g = B_h

外加一个逐 (层,头) 嵌入。头嵌入是**文档无关**的，所以它只能承载跨篇共享的成分；
而逐 chunk 的 `u` 已实测跨篇 Spearman ≈ 0（剔掉不可辨识列后 −0.005/+0.012/
−0.159/−0.052）。于是全部希望都压在 `rs, mgm` 这两个**文档相关**的标量上。

**留一篇交叉验证**：用 9 篇训练、第 10 篇检验。若留出篇上 R² ≤ 0，说明这套特征
学不出 `u` —— 那么无论训多少步、用什么网络，控制器都不可能复现教师的方向，
这条线就该收掉，而不是再烧 GPU 去训一遍再评一遍。

三个对照，缺一不可：
  ① 只用头嵌入（= 静态表上界）—— 分离「共享成分」与「上下文成分」；
  ② 打乱 `u`（保持设计矩阵）—— 给出 R² 的零分布；
  ③ 篇内留出（同篇不同 chunk）—— 若篇内能学而跨篇不能，问题是**泛化**不是表达力。
"""
import glob
import json
import os

import numpy as np
import torch
from scipy import stats as st

ROOT = os.path.dirname(os.path.abspath(__file__))


def fit_u(recs, lam=None, folds=5, seed=0):
    X = np.array([r["d"] for r in recs], dtype=np.float64)
    y = np.array([r["A"] for r in recs], dtype=np.float64)
    X = X - X.mean(0)
    sx = X.std() or 1.0
    Xn = X / sx

    def cv(l_):
        rs_ = np.random.default_rng(seed)
        idx = rs_.permutation(len(y))
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
    u = np.linalg.solve(Xn.T @ Xn + lams[i] * np.eye(Xn.shape[1]),
                        Xn.T @ (y - y.mean())) / sx
    return u - u.mean(), r2s[i], X


def main():
    # ---- 标签 ----
    R = []
    for f in sorted(glob.glob(os.path.join(ROOT, "scratch_labc_d*.json"))):
        R += [x for x in json.load(open(f)) if x.get("mode") == "chunk"]
    assert R, "没有 chunk 标签"

    rows = []
    for doc in sorted({r["doc"] for r in R}):
        tr_f = os.path.join(ROOT, f"scratch_ctrl_traces_v2/doc{doc:03d}.pt")
        if not os.path.exists(tr_f):
            print(f"  doc{doc} 无 trace，跳过")
            continue
        T = torch.load(tr_f, map_location="cpu", weights_only=False)
        L, H = T["L"], T["H"]
        for ci in sorted({r["chunk"] for r in R if r["doc"] == doc}):
            sub = [r for r in R if r["doc"] == doc and r["chunk"] == ci]
            if len(sub) < 40 or ci >= len(T["chunks"]):
                continue
            u, r2, X = fit_u(sub)
            sup = ((X > 0).sum(0) > 0) & ((X < 0).sum(0) > 0)   # 只有双边支撑列可解读
            ch = T["chunks"][ci]
            gsig = float(ch["gsig"])
            for l in range(L):
                lay = ch["layers"][l]
                tau = float(lay["thres"])
                for h in range(H):
                    g = l * H + h
                    if not sup[g]:
                        continue
                    sh = float(lay["sig_h"][h]); mh = float(lay["mu_h"][h])
                    rows.append(dict(doc=doc, chunk=ci, g=g, l=l, h=h,
                                     rs=np.log(max(sh, 1e-9) / max(gsig, 1e-9)),
                                     mgm=(mh - tau) / max(gsig, 1e-9),
                                     u=float(u[g]), r2=r2))
    assert rows, "join 后为空"
    docs = sorted({r["doc"] for r in rows})
    print(f"  join 得到 {len(rows)} 个 (篇,chunk,层,头) 样本，{len(docs)} 篇")

    # 每个 (篇,chunk) 内把 u 归一到单位范数：岭回归的整体尺度不可比
    key = {}
    for r in rows:
        key.setdefault((r["doc"], r["chunk"]), []).append(r)
    for v in key.values():
        n = np.linalg.norm([x["u"] for x in v]) or 1.0
        for x in v:
            x["un"] = x["u"] / n

    y = np.array([r["un"] for r in rows])
    doc_id = np.array([r["doc"] for r in rows])

    def design(kind):
        rs = np.array([r["rs"] for r in rows]); mg = np.array([r["mgm"] for r in rows])
        emb = np.zeros((len(rows), 112))
        emb[np.arange(len(rows)), [r["g"] for r in rows]] = 1.0
        ctx = np.stack([rs, mg, rs * mg, rs ** 2, mg ** 2,
                        np.ones(len(rows))], 1)
        if kind == "emb":     return emb
        if kind == "ctx":     return ctx
        if kind == "both":    return np.concatenate([emb, ctx], 1)
        raise ValueError(kind)

    def loo(kind, yy, lam=1.0):
        """留一篇交叉验证的 R²（在留出篇上算，篇内再中心化以消掉整体尺度）。"""
        pr = np.zeros_like(yy)
        X = design(kind)
        for d in docs:
            te = doc_id == d; tr = ~te
            w = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]),
                                X[tr].T @ yy[tr])
            pr[te] = X[te] @ w
        return 1 - ((yy - pr) ** 2).sum() / ((yy - yy.mean()) ** 2).sum(), pr

    print(f"\n════ 留一篇交叉验证：特征能否预测 u ════")
    rng = np.random.default_rng(0)
    ysh = y.copy(); rng.shuffle(ysh)
    for kind, name in (("emb", "只用逐(层,头)嵌入（= 静态表上界）"),
                       ("ctx", "只用上下文特征 rs, mgm 及其二次项"),
                       ("both", "嵌入 + 上下文特征")):
        best = max((loo(kind, y, l)[0], l) for l in (0.01, 0.1, 1.0, 10.0, 100.0))
        sh = max(loo(kind, ysh, l)[0] for l in (0.01, 0.1, 1.0, 10.0, 100.0))
        print(f"  {name:34s} LOO R² = {best[0]:+.4f} (λ={best[1]:g})   打乱对照 {sh:+.4f}")

    # 篇内留出：同篇不同 chunk（分辨「表达力不足」与「泛化不了」）
    print(f"\n════ 篇内留出（同篇不同 chunk）：分辨表达力 vs 泛化 ════")
    chunk_id = np.array([f"{r['doc']}_{r['chunk']}" for r in rows])
    uq = sorted(set(chunk_id))
    for kind, name in (("ctx", "只用上下文特征"), ("both", "嵌入 + 上下文")):
        X = design(kind); pr = np.zeros_like(y)
        for c in uq:
            te = chunk_id == c
            tr = (doc_id == rows[int(np.where(te)[0][0])]["doc"]) & ~te
            if tr.sum() < 50:
                pr[te] = 0.0; continue
            w = np.linalg.solve(X[tr].T @ X[tr] + 1.0 * np.eye(X.shape[1]), X[tr].T @ y[tr])
            pr[te] = X[te] @ w
        r2 = 1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  {name:34s} 篇内留出 R² = {r2:+.4f}")

    print(f"\n  **判据**：留一篇 R² ≤ 0 ⇒ 这套特征学不出 u，控制器不可能复现教师方向；")
    print(f"    0 < R² < 0.1 ⇒ 有微弱信号，但下游多半读不出；R² ≥ 0.2 ⇒ 值得训。")
    print(f"  ⚠ 篇内能学而跨篇不能 ⇒ 问题是**泛化**不是表达力，换更大网络没用。")


if __name__ == "__main__":
    main()
