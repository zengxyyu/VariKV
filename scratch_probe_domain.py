#!/usr/bin/env python3
"""竞争域定理：保序重标定能否改变 selection，只取决于阈值方案暴露多少配额自由度。

**定理**  设每个组 g（这里 g = (层, KV头)）内的重标定 `T_g` 严格单调。把这些组按
阈值方案划分为若干**竞争域** `C_m`（同一域内的条目共享一个预算池）。则

    |C_m| = 1  ⇒  该域内 selection 与 T 无关（严格 no-op）
    |C_m| > 1  ⇒  T **可能**改变域内各组的配额，但组内排序恒不变

*证明（|C_m|=1）*：域内只有一个组，预算 b 由代码固定（`k = int(n·ρ)`），而
`TopK_k(T(s)) = TopK_k(s)` 由严格单调性得到。∎
*证明（|C_m|>1）*：见 `scratch_probe_quota.py` 的等价定理——全局 Top-B 的结果
等于 `∪_g Top_{b_g}(s_g)`，而 `b_g` 可随 T 改变。∎

配额自由度（本 harness，28 层 × 4 KV 头 = 112 组，逐行核对 `attention/score.py`）：

    pair-head    `_threshold_head`   k=int(n_seq*ρ) 逐头 topk   112 个单元素域   0
    adakv-layer  `_threshold_layer`  逐层 score.reshape(-1)      28 个 4 元素域   28×3 = 84
    pair         `_threshold`        全局 score.reshape(-1)      1 个 112 元素域  111

本探针在**真实 teacher trace 分数**与**真实训练好的 scalar 臂**上逐位验证这三行。
零 GPU。这同时是一个**阴性对照**：若 pair-head 下测出掩码有变化，说明要么该臂在
测到的 z 范围内不保序，要么定理的前提被违反——两种情况都必须在下结论前查清。
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
from attention.score import KVScore                            # noqa: E402


def load(path):
    sd = torch.load(path, map_location="cpu")
    m = CalibScorer(sd.get("d_kv", 128), sd["L"], sd["H"],
                    n_slots=sd.get("slots", 8), d_m=sd.get("dim", 128),
                    mode="memoryless", arch=sd["arch"])
    m.load_state_dict(sd["state"])
    return m.eval(), sd["arch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="scalar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traces", default="scratch_ctrl_traces_v2_10")
    ap.add_argument("--n_doc", type=int, default=3)
    ap.add_argument("--ratio", type=float, default=0.2)
    a = ap.parse_args()

    m, arch = load(os.path.join(ROOT, f"varikv/d10_{a.arm}_s{a.seed}.pt/memoryless.pt"))
    print(f"臂 {a.arm}  arch={arch}  参数 {m.n_params():,}  α={float(m.alpha):.4f}")
    sm = KVScore()

    # 逐 chunk 累计。score 组织成 list[层] of [H, n]，与 harness 一致。
    LV = ["pair-head", "adakv-layer", "pair"]
    diff = {L: [0, 0] for L in LV}        # [掩码不同的位数, 总位数]
    chg = {L: [0, 0] for L in LV}         # [配额改变的组数, 总组数]
    mov = {L: [0.0, 0.0] for L in LV}     # [Σ|Δb|/2 实际搬动量, Σb 总预算]
    inv_bad = inv_tot = 0                  # 组内逆序对
    for f in sorted(glob.glob(os.path.join(ROOT, a.traces, "doc*.pt")))[:a.n_doc]:
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            g, t = float(ch["gsig"]), float(ch["thres"])
            S0, SP = [], []
            for l, pl in enumerate(ch["layers"]):
                n = pl["n_near"]
                s0 = pl["s0"][:, :n].float()
                st = (pl["mu_h"].float(), pl["sig_h"].float(), torch.tensor(g))
                with torch.no_grad():
                    ds = m.delta(
                        m.feat(m.raw(pl["k"][:, :n].float(), pl["v"][:, :n].float()))
                        if arch in ("kv", "k", "v") else None,
                        m.read(m.init_state(l), None),
                        s0, margin=(s0 - t) / g, stats=st)
                S0.append(s0); SP.append(s0 + ds)
                # 组内逆序对：保序性的直接实测，不依赖网格
                for h in range(s0.shape[0]):
                    o = torch.argsort(s0[h]); dd = (s0[h] + ds[h])[o]
                    inv_bad += int((dd[1:] < dd[:-1]).sum()); inv_tot += len(o) - 1
            A = torch.stack(S0)[:, None]      # [L,1,H,n] —— threshold 会 squeeze(1)
            B = torch.stack(SP)[:, None]
            for L in LV:
                va, _ = sm.threshold(A, a.ratio, L)
                vb, _ = sm.threshold(B, a.ratio, L)
                diff[L][0] += int((va ^ vb).sum()); diff[L][1] += va.numel()
                qa = va.reshape(-1, va.shape[-1]).sum(-1)
                qb = vb.reshape(-1, vb.shape[-1]).sum(-1)
                chg[L][0] += int((qa != qb).sum()); chg[L][1] += qa.numel()
                mov[L][0] += float((qa - qb).abs().sum()) / 2.0
                mov[L][1] += float(qa.sum())

    print(f"\n组内逆序对 {inv_bad}/{inv_tot} = {inv_bad/max(inv_tot,1)*100:.6f}%"
          f"   （0 ⇒ 该臂在实测分数上严格保序，定理前提成立）")
    print(f"\nρ={a.ratio}，{a.n_doc} 篇 trace")
    print(f"{'level':<14}{'DOF':>6}{'掩码翻转位':>13}{'翻转率':>10}"
          f"{'配额变的组':>13}{'搬动量 Σ|Δb|/2':>16}{'占总预算':>10}")
    exp = {"pair-head": 0, "adakv-layer": 84, "pair": 111}
    for L in LV:
        b, tot = diff[L]; c, ct = chg[L]
        mv, bt = mov[L]
        print(f"{L:<14}{exp[L]:>6}{b:>13,}{b/max(tot,1)*100:>9.4f}%"
              f"{c:>7,}/{ct:<5,}{mv:>16,.0f}{mv/max(bt,1)*100:>9.3f}%")
    print("""
判读
  · pair-head 一行**必须**是 0 位翻转、0 个组配额改变 —— 这是定理的阴性对照，
    非零就说明保序性被违反（查逆序对那一行）或测法有误，全表作废。
  · adakv-layer 与 pair 非零 ⇒ 竞争域 >1 时保序重标定确实能重分配配额。
  · 两者的相对大小刻画「跨层竞争额外买到多少」：pair 比 adakv-layer 多出的部分
    只能来自跨层，因为层内自由度两者相同。
  · 数据限制：trace 每 (chunk,层,头) 只存 768 个候选，这里是**子总体**上的
    Top-B，绝对翻转率不可外推到推理时；三行同口径，相对比较有效。""")


if __name__ == "__main__":
    raise SystemExit(main())
