#!/usr/bin/env python3
"""保序重标定 ≡ 逐头配额分配：在真实数据上核验，并量化配额的上下文依赖性。

**定理（Rank-Preserving Calibration–Allocation Equivalence）**
设每个头 h 的重标定 `T_h` 在该头内严格单调（`∂T_h(s)/∂s > 0`）。对
`{T_h(s_{h,i})}` 执行全局 Top-B，则存在唯一（忽略平局）配额向量
`b = (b_1,…,b_H)`，`Σ_h b_h = B`，使

    Top_B({T_h(s_{h,i})})  ==  ∪_h Top_{b_h}(s_h)                      (1)

*证明*：设 S 为 Top-B 选出的集合，`b_h = |S ∩ h|`。若 `i ∈ S∩h`、`j ∉ S`、
`j ∈ h` 而 `s_{h,j} > s_{h,i}`，由单调性 `T_h(s_{h,j}) > T_h(s_{h,i})`，则 j
应先于 i 入选，矛盾。故 `S∩h` 恰为 `s_h` 的前 `b_h` 名。∎

**逆命题（满射）**：给定任意 `b`，取 `δ_h = T − s_{h,(b_h)}`（`s_{h,(b_h)}` 是
该头第 `b_h` 大的分数，T 为任一公共常数），则对 `s + δ` 做全局 Top-B 恰得 `b`。
⇒ **逐头常数平移已经用尽了全部保序重标定所能表达的决策。**

两条推论，正是本探针要测的：

1. `scalar` 族在头内是 `s⁰` 的一元函数（特征 z / mg / rs / e_h 全由 `s⁰` 与
   头-chunk 常量决定），且 `scratch_probe_monotone.py` 已给出 896 组 × 4001 点
   的网格单调性证书 ⇒ (1) 应当**逐位成立**。这里在**真实候选分数**上直接验，
   把网格证书换成实测。
2. `kv` 臂的 `raw` 只吃 `feat(k,v)` 与 `e_h`，**看不到 `s⁰`** ⇒ 可以任意重排
   头内顺序 ⇒ (1) 应当**不成立**。这是阴性对照：若 kv 也逐位成立，说明测法有问题。

同时量化"配额改了多少、改得有多依赖上下文"：

    Δb_{c,h} = b_{c,h}^{arm} − b_{c,h}^{base}

    静态可解释份额 R²_static = 1 − Σ(Δb_{c,h} − Δb̄_{·,h})² / Σ(Δb_{c,h} − Δb̄)²

`R²_static → 1` 表示这次重分配其实是静态的（那 `bias` 臂就该拿到同样的分，
而它只有 +0.33）；`R²_static` 低则重分配主要是逐 chunk 的。

**数据限制（必须随结果一起报）**：teacher trace 每 (chunk,层,头) 只存 768 个
候选（近阈值 + 随机），所以这里的 Top-B 是在**子总体**上做的。定理对任何总体都
成立，所以等价性检验完全有效；但 `b_{c,h}` 的**绝对值**不是真实推理时的配额，
要拿真实配额得在评测里挂钩子导出 `valid`。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["bias", "affine", "scalar", "kv"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traces", default="scratch_ctrl_traces_v2_10")
    ap.add_argument("--n_doc", type=int, default=3)
    ap.add_argument("--ratio", type=float, default=0.1,
                    help="在存下来的候选子总体上取的保留比例")
    a = ap.parse_args()

    ms = {}
    for A in a.arms:
        p = os.path.join(ROOT, f"varikv/d10_{A}_s{a.seed}.pt/memoryless.pt")
        if not os.path.exists(p):
            print(f"  跳过 {A}（缺 {p}）")
            continue
        ms[A] = load(p)
        print(f"  {A:<7} arch={ms[A][1]:<7} 参数 {ms[A][0].n_params():>7,}")

    # 每个 chunk 独立处理 —— prune_chunk 只对当前 chunk 的 evict_range 定阈值，
    # 旧决策从不回溯，所以预算是**逐 chunk** 分配的，不是整个上下文一次分配。
    eq = {A: [0, 0] for A in ms}          # [逐位相同的 chunk 数, 总 chunk 数]
    inv = {A: [] for A in ms}             # 头内逆序对比例
    DB = {A: [] for A in ms}              # (doc, chunk, layer, head, Δb)
    for f in sorted(glob.glob(os.path.join(ROOT, a.traces, "doc*.pt")))[:a.n_doc]:
        d = torch.load(f, map_location="cpu")
        doc = os.path.basename(f)
        for ci, ch in enumerate(d["chunks"]):
            g, t = float(ch["gsig"]), float(ch["thres"])
            S0, HID, KK, VV, MG, ST = [], [], [], [], [], []
            for l, pl in enumerate(ch["layers"]):
                n = pl["n_near"]
                s0 = pl["s0"][:, :n].float()
                H = s0.shape[0]
                S0.append(s0); KK.append(pl["k"][:, :n].float())
                VV.append(pl["v"][:, :n].float())
                MG.append((s0 - t) / g)
                ST.append((pl["mu_h"].float(), pl["sig_h"].float(), torch.tensor(g)))
                HID.append(torch.full((H, n), l * H, dtype=torch.long)
                           + torch.arange(H)[:, None])
            hid = torch.cat([x.reshape(-1) for x in HID])
            s0f = torch.cat([x.reshape(-1) for x in S0])
            B = max(int(len(s0f) * a.ratio), 1)

            def topB(x):
                m_ = torch.zeros(len(x), dtype=torch.bool)
                m_[torch.topk(x, B).indices] = True
                return m_
            base = topB(s0f)
            nb = torch.bincount(hid[base], minlength=int(hid.max()) + 1)

            for A, (m, arch) in ms.items():
                with torch.no_grad():
                    ds = torch.cat([
                        m.delta(m.feat(m.raw(KK[l], VV[l])) if arch in ("kv", "k", "v")
                                else None,
                                m.read(m.init_state(l), None),
                                S0[l], margin=MG[l], stats=ST[l]).reshape(-1)
                        for l in range(len(S0))])
                sel = topB(s0f + ds)
                q = torch.bincount(hid[sel], minlength=len(nb))
                # 配额重放：每个头按 s⁰ 原序取前 q_h 名
                rep = torch.zeros_like(sel)
                for h in torch.unique(hid):
                    if q[h] == 0:
                        continue
                    idx = (hid == h).nonzero(as_tuple=True)[0]
                    rep[idx[torch.topk(s0f[idx], int(q[h])).indices]] = True
                eq[A][0] += int(bool(torch.equal(rep, sel))); eq[A][1] += 1
                # 头内逆序对比例（保序性的直接度量，不依赖网格）
                bad = tot = 0
                for h in torch.unique(hid):
                    idx = (hid == h).nonzero(as_tuple=True)[0]
                    o = torch.argsort(s0f[idx])
                    dd = (s0f + ds)[idx][o]
                    bad += int((dd[1:] < dd[:-1]).sum()); tot += len(idx) - 1
                inv[A].append(bad / max(tot, 1))
                for h in range(len(nb)):
                    DB[A].append((doc, ci, h, int(q[h]) - int(nb[h])))

    print(f"\n{'臂':<8}{'配额重放逐位相同':>18}{'头内逆序对':>12}"
          f"{'|Δb| 均值':>12}{'R²_static':>12}")
    for A in ms:
        rows = DB[A]
        hs = np.array([r[2] for r in rows]); db = np.array([r[3] for r in rows], float)
        # 静态可解释份额：只用"逐头一个常数"能解释 Δb 多少方差
        pred = np.array([db[hs == h].mean() for h in hs])
        ss_r = ((db - pred) ** 2).sum(); ss_t = ((db - db.mean()) ** 2).sum()
        r2 = 1 - ss_r / ss_t if ss_t > 0 else float("nan")
        print(f"{A:<8}{eq[A][0]:>10}/{eq[A][1]:<7}{np.mean(inv[A])*100:>11.4f}%"
              f"{np.abs(db).mean():>12.2f}{r2:>12.4f}")

    print("""
判读
  · 配额重放逐位相同 = 定理 (1) 在真实数据上成立。scalar/bias/affine 应当全中；
    kv 应当不中（它看不到 s⁰，可以重排头内顺序）—— 那是阴性对照，全中反而是 bug。
  · 头内逆序对 0 ⇒ 保序，把网格证书换成了实测。
  · R²_static 是「一个逐头常数能解释多少配额改动」。它高 ⇒ 重分配基本是静态的，
    那 bias 臂就该拿到同样的下游分；实测 bias 只有 +0.33 ± 0.50 而 scalar +4.73 ± 0.41,
    所以预期 R²_static 明显小于 1。
  · 绝对配额不可外推：trace 每 (chunk,层,头) 只存 768 个候选，这里的 Top-B 在子总体上。""")


if __name__ == "__main__":
    raise SystemExit(main())
