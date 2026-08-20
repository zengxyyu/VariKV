#!/usr/bin/env python3
"""无 σ 缩放的统一界 `C*`：地板配额要多大的均匀幅度才可达 —— 零 GPU。

**为什么这是决定方法方向的那一枪。** 当前修正的界是 `|Δs_h| ≤ α·σ_h`，
**与该头自己的分数离散度绑死**；已测 κ ≥ 11×（要把这个界整体放大 11 倍
地板配额才进入可达集）。外部复核提出的问题是：**瓶颈是维度不够，还是
幅度被错误地耦合到 σ_h 上？** 换成**统一界** `|c_h| ≤ C` 就能分辨。

闭式解（比 κ 那次的二分更干净）：统一界下
    slack(C) = min_h[s_(q_h) + C] − max_h[s_(q_h+1) − C]
             = min_h s_(q_h) − max_h s_(q_h+1) + 2C
⇒ 可达 ⟺ **C > (max_h s_(q_h+1) − min_h s_(q_h)) / 2**，即
    C* = max(0, (max_h s_(q_h+1) − min_h s_(q_h)) / 2).

判据：把 C* 与 ①逐头界 α·σ_h 的分位数、②全局分数尺度 σ_g 比。
若 **C* 落在 α·σ_h 的分布之内**（即「有些头本来就允许这么大的幅度，
只是需要这个幅度的头不被允许」）⇒ **病灶是尺度耦合，不是幅度不够**。
"""
import glob, os, sys
import numpy as np, torch
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.calib_scorer import CalibScorer                       # noqa: E402

H = 4


def main():
    bmin = float(os.environ.get("BMIN", "1"))
    sd = torch.load(f"{ROOT}/varikv/d10_scalar_s0.pt/memoryless.pt", map_location="cpu")
    m = CalibScorer(sd.get("d_kv", 128), sd["L"], sd["H"], n_slots=sd.get("slots", 8),
                    d_m=sd.get("dim", 128), mode="memoryless", arch=sd["arch"])
    m.load_state_dict(sd["state"]); m.eval()
    alpha = float(m.alpha)

    rows, skipped = [], 0
    for fp in sorted(glob.glob(f"{ROOT}/scratch_ctrl_traces_v2_10/doc*.pt")):
        d = torch.load(fp, map_location="cpu")
        for ch in d["chunks"]:
            t = float(ch["thres"])
            S0, SIG = [], []
            for pl in ch["layers"]:
                n = pl["n_near"]
                S0.append(pl["s0"][:, :n].float()); SIG.append(pl["sig_h"].float())
            sc = torch.cat(S0, 0); sig = torch.cat([x.reshape(-1) for x in SIG])
            G, npt = sc.shape
            s0f = sc.reshape(-1)
            B = int((s0f > t).sum())
            if B < G or B >= len(s0f) - G:
                skipped += 1; continue
            vb = torch.zeros(len(s0f), dtype=torch.bool)
            vb[torch.topk(s0f, B).indices] = True
            b0 = vb.reshape(G, npt).sum(-1).float()
            # 地板目标配额（与 floorcov f=1 同构：饿死头抬到 bmin，富余头按比例扣回）
            tg = torch.maximum(b0, torch.full_like(b0, bmin))
            ex = float(tg.sum() - B)
            if ex > 0:
                room = (b0 - bmin).clamp(min=0); tr = float(room.sum())
                if tr < ex: skipped += 1; continue
                tg = tg - room * (ex / tr)
            tg = torch.round(tg).long().clamp(0, npt)
            dfz = int(tg.sum()) - B
            if dfz != 0:
                idx = torch.argsort(-b0)
                for k in range(abs(dfz)):
                    tg[int(idx[k % G])] -= int(np.sign(dfz))
            SQ = np.empty(G); SQ1 = np.empty(G)
            for h in range(G):
                sh = np.sort(sc[h].numpy())[::-1]
                q = int(tg[h])
                SQ[h] = sh[q - 1] if q >= 1 else np.inf
                SQ1[h] = sh[q] if q < len(sh) else -np.inf
            fin_q = SQ[np.isfinite(SQ)]; fin_q1 = SQ1[np.isfinite(SQ1)]
            if not len(fin_q) or not len(fin_q1): skipped += 1; continue
            cstar = max(0.0, (fin_q1.max() - fin_q.min()) / 2.0)
            a = alpha * sig.numpy()
            rows.append((cstar, np.median(a), a.max(), np.percentile(a, 90),
                         float(sc.std())))
    A = np.array(rows)
    print(f"{len(rows)} 个 chunk（跳过 {skipped}）  地板 b_min={bmin:.0f}  α={alpha:.6f}")
    print(f"\n【统一界 C*】使地板配额可达的最小均匀幅度")
    print(f"  C*        中位 {np.median(A[:,0]):.4f}   p10 {np.percentile(A[:,0],10):.4f}"
          f"   p90 {np.percentile(A[:,0],90):.4f}")
    print(f"\n【对比：当前逐头界 α·σ_h 的分布】")
    print(f"  中位 α·σ_h  {np.median(A[:,1]):.4f}    p90 {np.median(A[:,3]):.4f}"
          f"    **最大 α·σ_h {np.median(A[:,2]):.4f}**")
    print(f"  分块内分数标准差 σ_g  {np.median(A[:,4]):.4f}")
    r_med = A[:, 0] / A[:, 1]; r_max = A[:, 0] / A[:, 2]
    print(f"\n【判词由数字生成】")
    print(f"  C* / 中位 α·σ_h   中位 **{np.median(r_med):.2f}×**")
    print(f"  C* / 最大 α·σ_h   中位 **{np.median(r_max):.2f}×**"
          f"   （<1 表示**已经有头被允许这么大的幅度**）")
    inside = float((A[:, 0] <= A[:, 2]).mean())
    print(f"  C* ≤ 该 chunk 的最大 α·σ_h 的比例 **{100*inside:.1f}%**")
    print(f"\n  ⇒ {'**病灶是尺度耦合**：所需幅度落在现有逐头界的分布之内，'
             '只是需要它的头不被允许' if inside > 0.5 else
             '**所需幅度超出现有任何头的界**，不只是耦合问题，幅度本身也不够'}")


if __name__ == "__main__":
    raise SystemExit(main())
