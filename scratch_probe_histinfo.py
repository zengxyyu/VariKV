#!/usr/bin/env python3
"""历史到底有没有增量预测价值 —— 用**凸**模型问，绕开全部优化风险。

B 至今每一次零结果都可以归咎于优化，而不是信息：
    α 太小（Δs 满幅只有典型 |Δs0| 的 12%，只有 24% 的成对翻得动）；
    GRU 把方向压掉（合成阶梯 base −0.0015）；
    读出的 softmax 凸组合表达不了有符号对比（C→D 掉 68%）；
    单次训练的方差（v1 的 +21.60 三次重训跨度 39 分）。
这些都是"模型学不到"，不是"信息不存在"。两者混着，永远得不出结论。

所以这里问一个**线性**问题：在当前特征之外，把历史当额外特征加进一个线性排序器，
留出文档上的成对排序准确率会不会提高？线性 + 成对 logistic 损失是凸的，优化不可能
失败；它给出的是历史增量价值的**下界**——线性都能看见的东西，非线性模型只会更多。

三个模型，同一批特征标准化、同一划分、同一成对采样：
    (a) s0        原始门控分，不拟合
    (b) cur       仅当前特征（s0/z/margin/‖k‖/‖v‖…）
    (c) cur+hist  再加历史特征
`(c) − (b)` 就是 E[U|X,M] 与 E[U|X] 的线性可分辨差异。

历史特征只用 writer 在部署时真正看得到的东西：此前各 chunk 中**随机子样本**里
被保留/被驱逐候选的 [k;v] running mean。刻意不用近阈值那半——它是有偏的，且
推理时 writer 也不看它。
"""
import argparse
import glob
import os
import random

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.dirname(__file__))


def _mean(x, m):
    w = m.float()[..., None]
    return (x * w).sum(1) / w.sum(1).clamp_min(1.0)          # [H,d]


def _cos(a, b):
    """a [H,n,d] · b [H,d] → [H,n] 余弦。零向量给 0。"""
    return F.cosine_similarity(a, b[:, None, :].expand_as(a), dim=-1)


def build(files, rho, dev, nbank=48):
    """→ (feat_cur [N,Fc], feat_hist [N,Fh], U [N], grp [N]) ，grp = 采样成对时的组 id。"""
    Xc, Xh, Us, Gs = [], [], [], []
    gid = 0
    for fi, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        run = {}                                              # layer → (mR, mE) [H,2d]
        for ci, ch in enumerate(d["chunks"]):
            gsig = float(ch.get("gsig", 1.0))
            for l, pl in enumerate(ch["layers"]):
                nn_ = pl["n_near"]
                k, v = pl["k"].float(), pl["v"].float()
                kv = torch.cat([k, v], -1)                    # [H,768,2d]
                s0 = pl["s0"].float()
                ret = pl["ret"]
                H = s0.shape[0]
                prev = run.get(l)
                if ci > 0 and prev is not None:
                    mR, mE, bR, bE = prev
                    kn = kv[:, :nn_]
                    # ---- 一阶（方向）：与已保留/已驱逐**均值**的余弦 ----
                    hm = [_cos(kn, mE), _cos(kn, mR), _cos(kn, mE - mR)]
                    # ---- 冗余（覆盖度）：与此前**具体**保留/驱逐条目的最大相似度 ----
                    # 让目标非模块化的机制是冗余——"这个 token 是不是和某个已经留下的
                    # 重复"——那是 **max** 而不是 mean。均值对覆盖度是很差的代理：
                    # 一堆互相正交的已保留键，其均值范数接近 0，与任何候选的余弦都≈0，
                    # 可覆盖度其实很高。次模性说的正是这件事，所以必须单独测。
                    kdim = k.shape[-1]
                    knk = F.normalize(kn[..., :kdim], dim=-1)
                    def _mx(bank):
                        if bank is None or bank.shape[1] == 0:
                            z = torch.zeros(kn.shape[0], kn.shape[1])
                            return z, z
                        b = F.normalize(bank[..., :kdim], dim=-1)      # [H,m,d]
                        sim = torch.einsum("hnd,hmd->hnm", knk, b)
                        top = sim.topk(min(5, sim.shape[-1]), dim=-1).values
                        return top[..., 0], top.mean(-1)
                    mxR, t5R = _mx(bR)
                    mxE, t5E = _mx(bE)
                    hist = torch.stack(hm + [mxR, t5R, mxE, mxR - mxE], dim=-1)
                    mu = pl["mu_h"].float()[:, None]
                    sg = pl["sig_h"].float()[:, None].clamp_min(1e-6)
                    s0n = s0[:, :nn_]
                    cur = torch.stack([
                        (s0n - mu) / sg,                       # 头内 z
                        (s0n - float(pl["thres"])) / gsig,     # 到全局阈值
                        k[:, :nn_].norm(dim=-1),
                        v[:, :nn_].norm(dim=-1),
                        s0n,
                    ], dim=-1)                                 # [H,nn,5]
                    # **逐 (chunk,层,头) 标准化**：单个全局线性模型没法适应各组尺度差异，
                    # 不归一的话它会去学尺度而不是内容。
                    for T in (cur, hist):
                        T -= T.mean(1, keepdim=True)
                        T /= T.std(1, keepdim=True).clamp_min(1e-6)
                    Xc.append(cur.reshape(-1, cur.shape[-1]))
                    Xh.append(hist.reshape(-1, hist.shape[-1]))
                    Us.append(pl["U"].float()[:, :nn_].reshape(-1))
                    Gs.append(torch.full((H * nn_,), gid, dtype=torch.long)
                              + torch.arange(H).repeat_interleave(nn_))
                    gid += H
                # ---- 更新 running mean（只用随机子样本，与 writer 部署时一致）----
                kr, rr = kv[:, nn_:], ret[:, nn_:]
                cR, cE = _mean(kr, rr), _mean(kr, ~rr)
                # 冗余特征需要**具体条目**而不只是均值。逐头保留的条数不同，
                # 所以按头各取固定 nbank 条（不足则补齐已有的），维持矩形张量。
                def _bank(mask):
                    out = []
                    for h in range(kr.shape[0]):
                        sel = kr[h][mask[h]]
                        if sel.shape[0] > nbank:
                            sel = sel[torch.randperm(sel.shape[0])[:nbank]]
                        elif sel.shape[0] == 0:
                            sel = torch.zeros(1, kr.shape[-1])
                        out.append(sel[:nbank])
                    m_ = min(o_.shape[0] for o_ in out)
                    return torch.stack([o_[:m_] for o_ in out])
                nR, nE = _bank(rr), _bank(~rr)
                if prev is None:
                    run[l] = (cR, cE, nR, nE)
                else:
                    keep = lambda old, new: torch.cat(  # noqa: E731
                        [old, new], 1)[:, -nbank * 4:]      # 只留最近 4 个 chunk 的量
                    run[l] = (rho * prev[0] + (1 - rho) * cR,
                              rho * prev[1] + (1 - rho) * cE,
                              keep(prev[2], nR), keep(prev[3], nE))
    return (torch.cat(Xc).to(dev), torch.cat(Xh).to(dev),
            torch.cat(Us).to(dev), torch.cat(Gs).to(dev))


def _design(X, U, G, n, gen, dev):
    """一次性构造成对设计矩阵：返回 (D=s·ΔX, w)。之后问题就是**确定性**的
    加权 logistic 回归，可以解到收敛。"""
    i = torch.randint(0, X.shape[0], (n,), generator=gen, device=dev)
    j = torch.randint(0, X.shape[0], (n,), generator=gen, device=dev)
    ok = (G[i] == G[j]) & ((U[i] - U[j]).abs() > 1e-9)   # 只在同 (chunk,层,头) 内比
    i, j = i[ok], j[ok]
    du = U[i] - U[j]
    w = (du.abs() / du.abs().median().clamp_min(1e-12)).clamp(0, 5)
    return (X[i] - X[j]) * du.sign()[:, None], w


def fit_eval(Xtr, Utr, Gtr, Xva, Uva, Gva, steps, n_pairs, seed, dev):
    """线性排序器 + 加权成对 logistic。**必须解到收敛**：模型是嵌套的
    （历史权重置零即退化成 cur），10 个参数对 150 万行不可能过拟合，所以
    "加了特征反而变差" 只能是没收敛。首版用 3000 步 Adam，跨种子 sd 达 0.034
    且 (c)<(b)，正是这个症状。改用固定设计矩阵 + LBFGS。"""
    g = torch.Generator(device=dev).manual_seed(seed)
    D, w = _design(Xtr, Utr, Gtr, n_pairs, g, dev)
    th = torch.zeros(Xtr.shape[1], device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([th], max_iter=steps, tolerance_grad=1e-10,
                            tolerance_change=1e-12, history_size=50,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (w * F.softplus(-(D @ th))).sum() / w.sum()
        loss.backward()
        return loss

    opt.step(closure)
    ltr = closure()                                  # 顺带拿到收敛后的梯度
    gn = float(th.grad.norm())
    with torch.no_grad():
        gv = torch.Generator(device=dev).manual_seed(7)   # 验证对固定，与 seed 无关
        Dv, wv = _design(Xva, Uva, Gva, 600000, gv, dev)
        lg = Dv @ th
        lva = float((wv * F.softplus(-lg)).sum() / wv.sum())
        return dict(acc=float(((lg > 0).float() * wv).sum() / wv.sum()),
                    acc_u=float((lg > 0).float().mean()),
                    th=th.detach(), ltr=float(ltr), lva=lva, gn=gn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="scratch_ctrl_traces_v2")
    ap.add_argument("--rho", type=float, default=0.5, help="历史 running mean 的 EMA 系数")
    ap.add_argument("--steps", type=int, default=500, help="LBFGS 最大迭代")
    ap.add_argument("--n_pairs", type=int, default=2000000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(os.path.join(ROOT, a.traces, "doc*.pt")))
    print(f"**留一交叉验证**，{len(files)} 篇文档   ρ={a.rho}   device={dev}")
    print("单次 8/2 划分不够：历史特征是逐 (文档,chunk,层) 共享的，独立观测数远小于\n"
          "行数，而留出只有 2 篇 ⇒ 文档级方差主导。LOO 给出 10 个配对估计。\n", flush=True)

    # 每篇单独建特征，之后按需拼接（避免重复读盘）
    per = [build([f], a.rho, dev) for f in files]
    print(f"每篇 {per[0][0].shape[0]:,} 行   当前 {per[0][0].shape[1]} 维 / "
          f"历史 {per[0][1].shape[1]} 维\n", flush=True)

    def cat(idx, hist):
        Xc = torch.cat([per[i][0] for i in idx])
        U = torch.cat([per[i][2] for i in idx])
        # 组 id 必须跨文档唯一，否则不同文档的候选会被配成一对
        G, off = [], 0
        for i in idx:
            G.append(per[i][3] + off); off += int(per[i][3].max()) + 1
        G = torch.cat(G)
        if hist:
            Xc = torch.cat([Xc, torch.cat([per[i][1] for i in idx])], 1)
        return Xc, U, G

    print(f"{'held-out':>9}{'(a) s0':>10}{'(b) cur':>10}{'(c) +hist':>11}"
          f"{'Δloss':>10}{'Δacc':>9}")
    dls, das = [], []
    for h in range(len(files)):
        tr = [i for i in range(len(files)) if i != h]
        row = {}
        for nm, hi in (("b", False), ("c", True)):
            Xt, Ut, Gt = cat(tr, hi)
            Xv, Uv, Gv = cat([h], hi)
            row[nm] = fit_eval(Xt, Ut, Gt, Xv, Uv, Gv, a.steps, a.n_pairs, 0, dev)
            del Xt, Ut, Gt, Xv, Uv, Gv
        # s0 基线（cur 的最后一列，逐组标准化后单调等价）
        Xv, Uv, Gv = cat([h], False)
        with torch.no_grad():
            gv = torch.Generator(device=dev).manual_seed(7)
            Dv, wv = _design(Xv, Uv, Gv, 600000, gv, dev)
            a0 = float(((Dv[:, -1] > 0).float() * wv).sum() / wv.sum())
        dl = row["c"]["lva"] - row["b"]["lva"]
        da = row["c"]["acc"] - row["b"]["acc"]
        dls.append(dl); das.append(da)
        print(f"{'doc%d' % h:>9}{a0:>10.4f}{row['b']['acc']:>10.4f}"
              f"{row['c']['acc']:>11.4f}{dl:>+10.5f}{da:>+9.4f}", flush=True)
        del Xv, Uv, Gv

    import statistics as st
    for nm, v, good in (("Δloss (负=历史有用)", dls, -1), ("Δacc  (正=历史有用)", das, +1)):
        m, sd = st.mean(v), st.stdev(v)
        t = m / (sd / len(v) ** 0.5)
        print(f"\n{nm}:  mean {m:+.5f} ± {sd:.5f}   n={len(v)}  t={t:+.2f}"
              f"   临界(df={len(v)-1},双侧95%)=2.262")
        if abs(t) <= 2.262:
            print("   → **与 0 不可分**")
        else:
            print(f"   → 显著，方向 {'支持' if m * good > 0 else '**不利于**'}历史假设")
    print("\n判读：Δloss 显著为负 ⇒ E[U|X,M] ≠ E[U|X]，B 的核心命题在数据层面成立。"
          "\n      不可分或为正 ⇒ 线性看不见历史信息；非线性仍可能有，但先验应大幅下调，"
          "\n      且此时继续调架构是没有依据的。")


if __name__ == "__main__":
    raise SystemExit(main())
