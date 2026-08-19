#!/usr/bin/env python3
"""有界残差下目标配额是否**可表示**？—— 规范不变的精确可行性判据，零 GPU。

**为什么要重做**：`scratch_probe_shift.py` 报的「所需 |c_h| 是 α·σ_h 的 13.3 倍」
**是规范假象**。逐头平移有规范自由度 `c → c + C·1`（全体同移不改变 Top-B），
而那个脚本把公共阈值固定成 `τ* = 0`，于是 `c_h ≈ −s_(q_h)` ——
量级是**分数本身**而不是**缺口**。比值随 `τ*` 任意变化，不可引用。**撤回 41。**

**正确的判据（规范不变）**。当前架构允许**逐 token** 的有界修正

    |Δs_{h,i}| ≤ α·σ_h              （不是只能整体平移！）

头 `h` 恰留 `q_h` 个，需要存在**公共阈值** `τ′` 使

    s_{h,(q_h)}   + Δs  >  τ′       最有利取 Δs = +α·σ_h
    s_{h,(q_h+1)} + Δs  ≤  τ′       最有利取 Δs = −α·σ_h

⇒ `τ′` 必须落在

    max_h [ s_{h,(q_h+1)} − α·σ_h ]  ≤  τ′  <  min_h [ s_{h,(q_h)} + α·σ_h ]

**该区间非空 ⟺ 目标配额在有界残差族下可表示。** 定义

    slack = min_h [ s_{h,(q_h)} + α·σ_h ] − max_h [ s_{h,(q_h+1)} − α·σ_h ]

`slack > 0` 可达；`slack ≤ 0` **不可达，且与网络学没学到无关**。
这与 `τ*` 的选择无关，因此是规范不变的。

同时报**纯平移**族的同一判据（把 `α·σ_h` 换成 0），作为对照：
纯平移的 slack 就是 `min_h s_{h,(q_h)} − max_h s_{h,(q_h+1)}`。
"""
import glob, os, sys
import numpy as np, torch
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.calib_scorer import CalibScorer                      # noqa: E402


def main():
    bmin = float(os.environ.get("BMIN", "8"))
    sd = torch.load(f"{ROOT}/varikv/d10_scalar_s0.pt/memoryless.pt", map_location="cpu")
    m = CalibScorer(sd.get("d_kv",128), sd["L"], sd["H"], n_slots=sd.get("slots",8),
                    d_m=sd.get("dim",128), mode="memoryless", arch=sd["arch"])
    m.load_state_dict(sd["state"]); m.eval(); alpha = float(m.alpha)

    rows = []
    for f in sorted(glob.glob(f"{ROOT}/scratch_ctrl_traces_v2_10/doc*.pt")):
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            t = float(ch["thres"]); S0, SIG, HID = [], [], []
            for l, pl in enumerate(ch["layers"]):
                n = pl["n_near"]; s0 = pl["s0"][:, :n].float(); H = s0.shape[0]
                S0.append(s0); SIG.append(pl["sig_h"].float())
                HID.append(torch.full((H,n), l*H, dtype=torch.long)+torch.arange(H)[:,None])
            s0f = torch.cat([x.reshape(-1) for x in S0]); hid = torch.cat([x.reshape(-1) for x in HID])
            sig = torch.cat([x.reshape(-1) for x in SIG])
            LH = int(hid.max())+1; npt = S0[0].shape[1]; B = int((s0f > t).sum())
            if B < 1 or B >= len(s0f): continue
            vb = torch.zeros(len(s0f), dtype=torch.bool); vb[torch.topk(s0f,B).indices]=True
            b0 = torch.bincount(hid[vb], minlength=LH).float()
            eff = min(bmin, B//LH)
            tg = torch.maximum(b0, torch.full_like(b0, eff)); ex = float(tg.sum()-B)
            if ex > 0:
                room=(b0-eff).clamp(min=0); tr=float(room.sum())
                if tr < ex: continue
                tg = tg - room*(ex/tr)
            tg = torch.round(tg).long().clamp(0, npt); dfz = int(tg.sum())-B
            if dfz != 0:
                idx = torch.argsort(-b0)
                for k in range(abs(dfz)): tg[int(idx[k%LH])] -= int(np.sign(dfz))
            SQ = np.empty(LH); SQ1 = np.empty(LH); AA = np.empty(LH)
            for h in range(LH):
                sh = np.sort(s0f[hid==h].numpy())[::-1]; q = int(tg[h])
                SQ[h]  = sh[q-1] if q >= 1 else  np.inf
                SQ1[h] = sh[q]   if q < len(sh) else -np.inf
                AA[h]  = alpha*float(sig[h])
            def slack(kappa):
                return float(np.min(SQ + kappa*AA) - np.max(SQ1 - kappa*AA))
            # 二分：界放大多少倍才刚好可行
            k = 1.0
            if slack(1.0) > 0: kk = 1.0
            else:
                hi_k = 1.0
                while slack(hi_k) <= 0 and hi_k < 1e6: hi_k *= 2
                lo_k = hi_k/2
                for _ in range(40):
                    mid = 0.5*(lo_k+hi_k)
                    if slack(mid) > 0: hi_k = mid
                    else: lo_k = mid
                kk = hi_k
            rows.append((slack(1.0), kk, 0, 0, float(b0.eq(0).float().mean())))
    A = np.array([r[:2] for r in rows])
    print(f"地板 b_min={bmin:.0f}  —— {len(rows)} 个 chunk（全部 10 篇 trace）")
    print(f"\n【有界残差族 |Δs| ≤ α·σ_h】（当前架构的真实表达能力）")
    print(f"  slack 中位 {np.median(A[:,0]):+.6f}   >0 的比例 **{(A[:,0]>0).mean()*100:.1f}%**")
    print(f"  ⇒ 地板配额在当前架构下{'**可表示**' if (A[:,0]>0).mean()>0.5 else '**不可表示**'}")
    print(f"\n【界要放大多少倍才刚好可行】（二分求 κ 使 slack(κ)=0）")
    print(f"  κ 中位 **{np.median(A[:,1]):.1f}×**   p10 {np.percentile(A[:,1],10):.1f}×"
          f"   p90 {np.percentile(A[:,1],90):.1f}×")
    print(f"  ⇒ 需要把 |Δs| ≤ α·σ_h 的界放大约 {np.median(A[:,1]):.0f} 倍，"
          f"地板配额才进入可表示集")


if __name__ == "__main__":
    raise SystemExit(main())
