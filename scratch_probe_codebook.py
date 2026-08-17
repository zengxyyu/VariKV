#!/usr/bin/env python3
"""码本退化探针：`M_init` 的 229,376 个参数里，有多少是真在起作用的？

**为什么问这个。** 读出是对 K+1 个槽做 softmax 注意力：

    r = softmax(q·Sᵀ/√d) @ S

实测（`d10_scalar` 之外的原版 v2，第 13 层）九个槽的平均注意力是
0.108 0.108 0.113 0.110 0.113 0.111 0.115 0.108 0.114 —— 几乎正好是 1/9。
如果注意力**逐 token 都接近均匀**，那 `r ≈ mean(S)`，是一个与 query 无关的常向量；
那样的话整个码本退化成"每 (层,头) 一个偏置"，229,376 个参数里真正起作用的远少于
名义值，而这恰恰是 `bias` 臂（225 参数）能表达的东西 —— 而 `bias` 拿到 +0.33。

**注意平均值接近 1/9 并不等于逐 token 均匀**：不同 token 各自尖锐地指向不同的槽，
平均下来一样是 1/9。所以必须逐 token 看熵，不能只看均值。

三个层次的度量，后一个比前一个更接近下游：

1. **注意力分布**：逐 token 的熵与最大权重，对比均匀分布的 log(K+1) 与 1/(K+1)。
2. **读出的 query 依赖性**：`r` 在 token 之间的变化幅度 vs 它自身的范数。
   `std_token(r)/‖mean(r)‖ ≈ 0` ⇒ 读出就是个常量。
3. **反事实消融（决定性的）**：把码本换成它自己的均值向量（= 强制均匀注意力的
   极限），重算 `Δs`，看
     · `Δs` 变了多少（相对 `Δs` 自身的标准差）
     · **有多少条候选会因此翻越阈值** —— 这才是下游真正在乎的量
   同时给一个上界参照：把读出整个置零（`r=0`）时翻转多少。

零 GPU：只用训练好的权重与教师 trace 里存的 K/V、s⁰、τ、逐头统计量。
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from varikv_v2 import V2Scorer                                      # noqa: E402


def delta_with(m, l, k, v, s0, margin, stats, mode="full"):
    """`mode`: full=原样 / mean=码本塌成自身均值 / zero=读出置零。"""
    xr = m.raw(k, v)
    x = m.x_proj(xr)
    q = m.q_read(xr)
    S_R, S_E = m.banks(l)
    if mode == "mean":
        # 均匀注意力的极限：softmax 权重全相等时 r 恒等于槽的均值
        S_R = S_R.mean(1, keepdim=True)
        S_E = S_E.mean(1, keepdim=True)
    if mode == "zero":
        rR = rE = torch.zeros(q.shape[0], q.shape[1], m.d_m)
    else:
        def _attn(S):
            a = torch.einsum("hnd,hkd->hnk", q, S) * m.d_m ** -0.5
            return torch.einsum("hnk,hkd->hnd", a.softmax(-1), S)
        rR, rE = _attn(S_R), _attn(S_E)
    mu_h, sig_h, sig_g = stats
    mu_h = mu_h.view(-1, 1); sig_h = sig_h.view(-1, 1).clamp_min(1e-6)
    z = (s0 - mu_h) / sig_h
    rs = (sig_h / sig_g).log().expand_as(z)
    sc = m.d_m ** -0.5
    raw = m.head(torch.cat(
        [x, rR, rE, rE - rR, q * rR, q * rE,
         (q * rR).sum(-1, keepdim=True) * sc, (q * rE).sum(-1, keepdim=True) * sc,
         z[..., None], margin[..., None], rs[..., None]], dim=-1)).squeeze(-1)
    return m.alpha * sig_h * torch.tanh(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="varikv/ctrl_b_a1_s0.pt/memoryless.pt")
    ap.add_argument("--traces", default="scratch_ctrl_traces_v2_10")
    ap.add_argument("--n_doc", type=int, default=2)
    a = ap.parse_args()

    m = V2Scorer.from_ckpt(os.path.join(ROOT, a.ckpt)).eval()
    K1 = m.M_init.shape[2] // 2 + 1                 # 每型槽数 = K + 1（D 槽）
    print(f"{a.ckpt}   每型 {K1} 个槽   均匀分布: 熵 {np.log(K1):.4f}, "
          f"最大权重 {1/K1:.4f}")

    ent, mx, rvar, rows = [], [], [], []
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(ROOT, a.traces, "doc*.pt")))[:a.n_doc]:
            d = torch.load(f, map_location="cpu")
            for ch in d["chunks"]:
                g, t = float(ch["gsig"]), float(ch["thres"])
                for l, pl in enumerate(ch["layers"]):
                    n = pl["n_near"]
                    k = pl["k"][:, :n].float(); v = pl["v"][:, :n].float()
                    s0 = pl["s0"][:, :n].float()
                    st = (pl["mu_h"].float(), pl["sig_h"].float(),
                          torch.tensor(g))
                    mg = (s0 - t) / g
                    q = m.q_read(m.raw(k, v))
                    S_R, _ = m.banks(l)
                    aw = (torch.einsum("hnd,hkd->hnk", q, S_R)
                          * m.d_m ** -0.5).softmax(-1)
                    ent.append(float((-aw * (aw + 1e-12).log()).sum(-1).mean()))
                    mx.append(float(aw.amax(-1).mean()))
                    r = torch.einsum("hnk,hkd->hnd", aw, S_R)
                    # 读出随 token 变化多少，相对它自身的平均范数
                    rvar.append(float(r.std(1).mean() / r.mean(1).norm(dim=-1).mean()))
                    df = delta_with(m, l, k, v, s0, mg, st, "full")
                    dm = delta_with(m, l, k, v, s0, mg, st, "mean")
                    dz = delta_with(m, l, k, v, s0, mg, st, "zero")
                    sd = float(df.std())
                    fl = lambda x, y: float(((s0 + x > t) ^ (s0 + y > t)).float().mean())
                    rows.append((sd, float((df - dm).std()), float((df - dz).std()),
                                 fl(df, dm), fl(df, dz),
                                 float(((s0 + df > t) ^ (s0 > t)).float().mean())))
    A = np.array(rows)
    print(f"\n【1. 注意力分布】逐 token 统计，{len(ent)} 个 (chunk,层)")
    print(f"  熵      均 {np.mean(ent):.4f}  最小 {np.min(ent):.4f}"
          f"   （均匀 = {np.log(K1):.4f}，越接近越退化）")
    print(f"  最大权重 均 {np.mean(mx):.4f}  最大 {np.max(mx):.4f}"
          f"   （均匀 = {1/K1:.4f}）")
    print(f"\n【2. 读出的 query 依赖性】std_token(r)/‖mean(r)‖ = {np.mean(rvar):.4f}"
          f"   （≈0 ⇒ 读出是常量）")
    print(f"\n【3. 反事实消融】相对 Δs 自身的标准差（{A[:,0].mean():.4f}）")
    print(f"  码本塌成均值:  Δ 变化 {A[:,1].mean():.4f}"
          f"  = {A[:,1].mean()/A[:,0].mean()*100:5.1f}% of std   "
          f"越阈值翻转 {A[:,3].mean()*100:.3f}%")
    print(f"  读出整个置零:  Δ 变化 {A[:,2].mean():.4f}"
          f"  = {A[:,2].mean()/A[:,0].mean()*100:5.1f}% of std   "
          f"越阈值翻转 {A[:,4].mean()*100:.3f}%")
    print(f"  （参照）方法本身相对基线的翻转率: {A[:,5].mean()*100:.3f}%")
    print("\n判读：塌成均值后翻转率≈0 ⇒ 码本退化成逐 (层,头) 的常量偏置，"
          "229,376 个槽参数\n      实际只提供 112 个自由度，与 `bias` 臂等价——"
          "而 bias 只拿到 +0.33。\n      翻转率与"
          "「方法本身的翻转率」同量级 ⇒ 码本的 query 依赖是真在起作用的。")


if __name__ == "__main__":
    raise SystemExit(main())
