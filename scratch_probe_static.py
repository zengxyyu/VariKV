#!/usr/bin/env python3
"""固定归一化偏好能否复现 `scalar` 的配额轨迹？—— 线性可行性判定，零 GPU。

**为什么需要这个探针：一个代码级更正。**

先前把 `bias` 说成"跨 chunk 固定的平移"，**错了**。`calib_scorer.py:157,172`：

    raw = r[:, 1:2]                       # 逐头一个常数
    Δs  = α · sig_h · tanh(raw)

而 `sig_h` 是**逐 (chunk, 层, 头)** 现算的（`scratch_ctrl_teacher.py:450`
`f0.std(-1)`）。所以 bias 的真实策略是

    Δs_{c,h} = σ_{c,h} · η_h ,      η_h = α·tanh(b_h)  跨上下文固定       (★)

即**固定的归一化偏好**，raw-score 平移量本身随 σ_{c,h} 变。于是"bias 只拿 +0.33
⇒ 静态重分配无效"这个推断**不成立**：bias 已经带 chunk 依赖，而且它的 +0.33 还
混着优化器有没有找到最优 η 的问题。

**正确的问法是结构性的，不是训练性的**：存在任何固定的 `{η_h}`（含最优的那个），
能复现 `scalar` 实际产生的配额轨迹 `b_{c,h}` 吗？

写成约束。设 chunk c 的全局阈值为 `T_c`（自由变量，因为阈值本来就是现算的），
`s⁰_{c,h,(j)}` 为该 (chunk,头) 内降序第 j 大的基线分数。要让头 h 恰好留下
`b_{c,h}` 个：

    s⁰_{c,h,(b)}   + σ_{c,h}·η_h  ≥  T_c          第 b 名必须过线
    s⁰_{c,h,(b+1)} + σ_{c,h}·η_h  <  T_c          第 b+1 名必须落榜        (1)

未知量只有 `{η_h}`（H 个）与 `{T_c}`（每 chunk 一个），(1) 对二者**都是线性的**。
加上代码强制的界 `|Δs| ≤ α·σ_h` ⇒ `η_h ∈ [−α, α]`，这是一个**线性可行性问题**。

    可行   ⇒ 固定归一化偏好足以复现整条轨迹（"静态够用"）
    不可行 ⇒ **不存在**任何固定归一化偏好能复现它，与训练无关 —— 这才是
             "静态策略结构上不够"的证明

**阳性对照是必须的**：对 `bias` 自己的配额轨迹跑同一个 LP，由 (★) 它**必然可行**
（`η_h = α·tanh(b_h)` 就是解）。若阳性对照不可行，说明 LP 构造有 bug，测试作废。

报最小总松弛（L1）。松弛以 σ_{c,h} 为单位归一化，跨头可比。

**数据限制**：teacher trace 每 (chunk,层,头) 只存 768 个候选，所以 `b_{c,h}` 是
**子总体**上的配额。可行性问题在该子总体上是良定义的、结论有效；但它不是推理时
的真实配额。
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

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


def collect(arms, seed, traces, n_doc, ratio):
    """→ per-arm 配额轨迹，外加每 (chunk,头) 的 σ 与降序基线分数。"""
    ms = {}
    for A in arms:
        p = os.path.join(ROOT, f"varikv/d10_{A}_s{seed}.pt/memoryless.pt")
        if os.path.exists(p):
            ms[A] = load(p)
            print(f"  {A:<7} arch={ms[A][1]:<7} α={float(ms[A][0].alpha):.3f} "
                  f"参数 {ms[A][0].n_params():>7,}")
    chunks = []                       # [{sig:[Hh], srt:[Hh][n], quota:{arm:[Hh]}}]
    for f in sorted(glob.glob(os.path.join(ROOT, traces, "doc*.pt")))[:n_doc]:
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            g, t = float(ch["gsig"]), float(ch["thres"])
            S0, KK, VV, MG, ST, HID, SIG = [], [], [], [], [], [], []
            for l, pl in enumerate(ch["layers"]):
                n = pl["n_near"]
                s0 = pl["s0"][:, :n].float(); H = s0.shape[0]
                S0.append(s0); KK.append(pl["k"][:, :n].float())
                VV.append(pl["v"][:, :n].float()); MG.append((s0 - t) / g)
                ST.append((pl["mu_h"].float(), pl["sig_h"].float(), torch.tensor(g)))
                HID.append(torch.full((H, n), l * H, dtype=torch.long)
                           + torch.arange(H)[:, None])
                SIG.append(pl["sig_h"].float().reshape(-1))
            hid = torch.cat([x.reshape(-1) for x in HID])
            s0f = torch.cat([x.reshape(-1) for x in S0])
            nH = int(hid.max()) + 1
            B = max(int(len(s0f) * ratio), 1)
            rec = {"sig": torch.cat(SIG).numpy(),
                   "srt": [np.sort(s0f[hid == h].numpy())[::-1] for h in range(nH)],
                   "quota": {}}
            for A, (m, arch) in ms.items():
                with torch.no_grad():
                    ds = torch.cat([
                        m.delta(m.feat(m.raw(KK[l], VV[l])) if arch in ("kv", "k", "v")
                                else None,
                                m.read(m.init_state(l), None),
                                S0[l], margin=MG[l], stats=ST[l]).reshape(-1)
                        for l in range(len(S0))])
                sel = torch.zeros(len(s0f), dtype=torch.bool)
                sel[torch.topk(s0f + ds, B).indices] = True
                rec["quota"][A] = torch.bincount(hid[sel], minlength=nH).numpy()
            chunks.append(rec)
    return ms, chunks


def feasible(chunks, arm, alpha, policy="norm"):
    """min Σ(u+v) s.t. (1) 带松弛。变量顺序 [η_1..η_H, T_1..T_C, u.., v..]。"""
    nH = len(chunks[0]["sig"]); nC = len(chunks)
    rows, cols, vals, rhs, kinds = [], [], [], [], []
    nu = 0                                        # 松弛计数
    slack_of = []
    for c, rec in enumerate(chunks):
        for h in range(nH):
            b = int(rec["quota"][arm][h]); srt = rec["srt"][h]
            # policy="norm": δ = σ_{c,h}·η_h（= bias 臂的真实策略类）
            # policy="raw" : δ = δ_h        （固定 raw-score 平移，另一类）
            sg = float(rec["sig"][h]) if policy == "norm" else 1.0
            # 下界：第 b 名过线   −σ·η + T − s_(b) − u ≤ 0
            if b >= 1:
                r = len(rhs)
                rows += [r, r, r]; cols += [h, nH + c, nH + nC + nu]
                vals += [-sg, 1.0, -1.0]; rhs.append(float(srt[b - 1])); kinds.append(1)
                slack_of.append(sg); nu += 1
            # 上界：第 b+1 名落榜   σ·η − T + s_(b+1) − v ≤ 0
            if b < len(srt):
                r = len(rhs)
                rows += [r, r, r]; cols += [h, nH + c, nH + nC + nu]
                vals += [sg, -1.0, -1.0]; rhs.append(-float(srt[b])); kinds.append(2)
                slack_of.append(sg); nu += 1
    nv = nH + nC + nu
    A_ub = coo_matrix((vals, (rows, cols)), shape=(len(rhs), nv)).tocsr()
    cvec = np.zeros(nv)
    # 松弛按 σ 归一化后求和 —— 否则大 σ 的头会主导目标
    cvec[nH + nC:] = 1.0 / np.maximum(np.array(slack_of), 1e-12)
    # raw 类的界是 |δ_h| ≤ α·min_c σ_{c,h}（要在所有 chunk 上都合法）
    if policy == "raw":
        mn = np.min(np.stack([r["sig"] for r in chunks]), axis=0)
        bnds = [(-alpha * float(mn[h]), alpha * float(mn[h])) for h in range(nH)]
    else:
        bnds = [(-alpha, alpha)] * nH
    bnds += [(None, None)] * nC + [(0, None)] * nu
    res = linprog(cvec, A_ub=A_ub, b_ub=np.array(rhs), bounds=bnds, method="highs")
    assert res.status in (0, 2), res.message
    if res.status == 2:
        return float("inf"), None, None
    slk = res.x[nH + nC:]
    viol = slk / np.maximum(np.array(slack_of), 1e-12)      # 以 σ 为单位
    return float(res.fun), viol, res.x[:nH]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["bias", "affine", "scalar", "kv"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traces", default="scratch_ctrl_traces_v2_10")
    ap.add_argument("--n_doc", type=int, default=3)
    ap.add_argument("--ratio", type=float, default=0.1)
    a = ap.parse_args()

    ms, chunks = collect(a.arms, a.seed, a.traces, a.n_doc, a.ratio)
    nH = len(chunks[0]["sig"])
    print(f"\n{len(chunks)} 个 chunk × {nH} 头；未知量 {nH} 个 η + {len(chunks)} 个 T_c")
    # 两块的松弛单位不同：norm 块除以 σ_{c,h}（σ 单位），raw 块不除（raw-score 单位）。
    # **两块之间的数值不可直接比较**，只能各自与本块的对照比。
    for pol, lab, u in (("norm", "δ = σ_{c,h}·η_h  归一化偏好（= bias 臂的真实策略类）", "σ"),
                        ("raw", "δ = δ_h          固定 raw-score 平移", "raw")):
        print(f"\n【{lab}】")
        print(f"{'臂':<8}{'总违反('+u+'单位)':>16}{'违反约束数':>12}{'最大单条':>12}{'判定':>14}")
        for A in a.arms:
            if A not in ms:
                continue
            obj, viol, eta = feasible(chunks, A, float(ms[A][0].alpha), pol)
            nbad = int((viol > 1e-7).sum()); mx = float(viol.max())
            ok = "**可行**" if obj < 1e-6 else "不可行"
            # 两个对照：norm 下 bias 必可行（η=α·tanh(b) 就是解）；
            # raw 下 bias 必**不**可行（它是归一化偏好，不是 raw 平移）。
            note = "  ← 阳性对照" if A == "bias" else ""
            print(f"{A:<8}{obj:>16.4f}{nbad:>12}{mx:>12.4f}{ok:>14}{note}")

    print("""
判读
  · 阳性对照 `bias` **必须**可行 —— 它的策略就是 Δs = σ_{c,h}·η_h，η_h = α·tanh(b_h)
    本身就是一个解。若它不可行，是 LP 构造有 bug，全表作废。
  · `scalar` 不可行 ⇒ **不存在**任何固定归一化偏好能复现它的配额轨迹。这是结构性
    结论，与优化器/训练轨迹无关，比"bias 训出来只有 +0.33"强得多。
  · `scalar` 可行 ⇒ 它的配额轨迹原则上静态可达，那 +4.73 vs +0.33 的差就要归到
    优化，而不是归到上下文依赖 —— 那样的话当前的机制叙事要大改。
  · 违反量以 σ_{c,h} 为单位：0.1 表示"差 0.1 个头标准差才能满足该约束"。
  · 数据限制：trace 每 (chunk,层,头) 只存 768 候选，b_{c,h} 是子总体配额。""")


if __name__ == "__main__":
    raise SystemExit(main())
