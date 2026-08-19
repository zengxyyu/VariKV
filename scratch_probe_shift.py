#!/usr/bin/env python3
"""逐头平移能否精确复现任意配额？—— 完备性定理的构造性检验，零 GPU。

**定理（外部复核提出，此处验证）**：固定预算的全局 Top-B 下，逐头平移
`s_{h,i} → s_{h,i} + c_h`
  (a) **精确保持**头内排序（差分 `T_i − T_j = s_i − s_j`，构造性）；
  (b) 无平局时**可实现任意合法配额** `q`（`Σq_h = B`）。

  证明 (b)：取任一公共阈值 `τ*`，头 h 恰留 `q_h` 个 ⟺
      τ* − s_{h,(q_h)} < c_h ≤ τ* − s_{h,(q_h+1)}
  因 `s_{h,(q_h)} > s_{h,(q_h+1)}`，该区间非空；各头独立取值即可。∎

  自由度：`c ∈ R^H`，但 `c → c + C·1` 不改变 Top-B（全体同移）⇒ 有效自由度 `H−1`。
  而配额单纯形 `{b ≥ 0, Σb = B}` 的维数同样是 `H−1`。**两者恰好相等。**

**本探针要回答的是更硬的一问**：地板（`b_h ≥ b_min`）给出的那个配额解，
需要多大的 `c_h`？若它远超当前参数化的界 `α·σ_h`，则

    Q_current ⊆ Q_feasible 是**严格**包含，且地板解落在外面

—— 这就把「可达性受限」从假说变成实测。
"""
import glob, os, sys
import numpy as np, torch
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.quota_project import project_quota                   # noqa: E402
from attention.calib_scorer import CalibScorer                      # noqa: E402


def solve_shift(sorted_desc, q, tau=0.0):
    """给定该头降序分数与目标配额 q，返回可行区间中点的 c_h。"""
    n = len(sorted_desc)
    hi = tau - sorted_desc[q] if q < n else np.inf          # c ≤ 上界
    lo = tau - sorted_desc[q-1] if q >= 1 else -np.inf      # c >  下界
    if lo == -np.inf: return hi - 1.0
    if hi == np.inf:  return lo + 1.0
    return 0.5*(lo+hi)


def main():
    bmin = float(os.environ.get("BMIN", "8"))
    sd = torch.load(f"{ROOT}/varikv/d10_scalar_s0.pt/memoryless.pt", map_location="cpu")
    m = CalibScorer(sd.get("d_kv",128), sd["L"], sd["H"], n_slots=sd.get("slots",8),
                    d_m=sd.get("dim",128), mode="memoryless", arch=sd["arch"])
    m.load_state_dict(sd["state"]); m.eval(); alpha = float(m.alpha)

    ok = tot = 0
    ratios = []          # |c_h| / (alpha*sigma_h)
    nz = 0; nz_big = 0
    for f in sorted(glob.glob(f"{ROOT}/scratch_ctrl_traces_v2_10/doc*.pt"))[:3]:
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            t = float(ch["thres"])
            S0, SIG, HID = [], [], []
            for l, pl in enumerate(ch["layers"]):
                n = pl["n_near"]; s0 = pl["s0"][:, :n].float(); H = s0.shape[0]
                S0.append(s0); SIG.append(pl["sig_h"].float())
                HID.append(torch.full((H,n), l*H, dtype=torch.long)+torch.arange(H)[:,None])
            s0f = torch.cat([x.reshape(-1) for x in S0])
            hid = torch.cat([x.reshape(-1) for x in HID])
            sig = torch.cat([x.reshape(-1) for x in SIG])
            LH = int(hid.max())+1; npt = S0[0].shape[1]
            B = int((s0f > t).sum())
            if B < 1 or B >= len(s0f): continue
            vb = torch.zeros(len(s0f), dtype=torch.bool); vb[torch.topk(s0f,B).indices]=True
            b0 = torch.bincount(hid[vb], minlength=LH).float()
            # 地板目标配额
            bt = project_quota(b0, torch.zeros_like(b0), npt, "floor", LH//4, 4) \
                 if False else None
            # 直接算：b_h ≥ min(bmin, B//LH)，缺口从富余头按比例扣回（与生产同逻辑）
            eff = min(bmin, B//LH)
            tg = torch.maximum(b0, torch.full_like(b0, eff))
            ex = float(tg.sum()-B)
            if ex > 0:
                room = (b0-eff).clamp(min=0); tr = float(room.sum())
                if tr < ex: continue
                tg = tg - room*(ex/tr)
            tg = torch.round(tg).long().clamp(0, npt)
            diff = int(tg.sum())-B
            if diff != 0:                       # 舍入修正
                idx = torch.argsort(-b0)
                for k in range(abs(diff)):
                    j = int(idx[k % LH]); tg[j] -= int(np.sign(diff))
            # 逐头解 c_h 并验证
            c = torch.zeros(LH)
            for h in range(LH):
                sh = np.sort(s0f[hid==h].numpy())[::-1]
                c[h] = solve_shift(sh, int(tg[h]))
                if b0[h] == 0 and tg[h] > 0:
                    nz += 1
                    r = abs(float(c[h]))/max(alpha*float(sig[h]), 1e-12)
                    ratios.append(r); nz_big += (r > 1)
            got = torch.bincount(hid[torch.topk(s0f + c[hid], B).indices], minlength=LH)
            ok += int(torch.equal(got, tg)); tot += 1
    print(f"地板 b_min={bmin:.0f}（实际按 min(b_min, B//112) 饱和）")
    print(f"  **配额逐位复现**：{ok}/{tot} 个 chunk 完全一致"
          f"   {'✓ 完备性定理在真实数据上成立' if ok==tot else '✗ 有不一致，需查平局'}")
    if ratios:
        r = np.array(ratios)
        print(f"\n  被地板救活的零配额头 n={nz}")
        print(f"  所需 |c_h| / (α·σ_h)：中位 **{np.median(r):.1f}×**  "
              f"p90 {np.percentile(r,90):.1f}×  最大 {r.max():.1f}×")
        print(f"  **超出当前参数化界（比值 >1）的比例：{nz_big/nz*100:.1f}%**")
        print(f"  ⇒ 地板解 {'落在' if nz_big/nz>0.5 else '未落在'} "
              f"Q_current 之外 ⇒ 包含关系是{'**严格的**' if nz_big/nz>0.5 else '未证严格'}")


if __name__ == "__main__":
    raise SystemExit(main())
