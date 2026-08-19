#!/usr/bin/env python3
"""饿死头能不能被「抬过线」？—— 架构可行性检查，零 GPU。

**为什么问这个**：Retr.KV@0.1 上零参数地板拿 +33.60（把 65.1% 的零配额头全部抬离
边界），而学到的方向只拿 +4.20。若网络**在物理上就抬不动**饿死头，那 +4.20 vs
+33.60 就不是 teacher 没教会，而是参数化本身封死了这条路。

一个头 h 在某 chunk 里「饿死」= 它的最高分仍低于全局阈值 τ：

    s_max(h) < τ        ⇒ b_h = 0

要让它拿到配额，必须把 s_max 抬过 τ。修正的形式是

    Δs = α · σ_h · tanh(φ(z)),      z = (s − μ_h)/σ_h

于是**可抬性**（closable fraction）：

    C_h = Δs(s_max) / (τ − s_max)          需要 ≥ 1 才能抬过线
    上界 U_h = α·σ_h / (τ − s_max)         tanh ≤ 1 给出的**物理上限**

**两条假说合并在这里**：
  假说 2（幅度不够）⇒ 看 `U_h`：若 U_h ≪ 1，**再怎么训也抬不动**。
  假说 3（方向反了）⇒ 看 `Δs(s_max)` 的**符号**：`d' ≤ 0` 是压上尾，
     若在 s_max 处 tanh(φ) < 0，网络实际在**把饿死头的顶端往下压**。

**数据限制**：trace 每 (chunk,层,头) 只存 768 个近阈值候选，所以 `s_max` 是
**候选池内**的最大值，可能低估真实头最大值 ⇒ `τ − s_max` 被高估、`C_h` 被低估。
因此**若测出 U_h ≫ 1（能抬动），结论稳**；若测出 U_h ≪ 1，需谨慎（可能是采样偏差）。
候选池按 |s⁰−τ| 采样，近阈值的被优先保留，所以对饿死头而言 s_max 的低估应当有限。
"""
import glob, os, sys
import numpy as np, torch
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.calib_scorer import CalibScorer                      # noqa: E402


def main():
    ck = os.path.join(ROOT, "varikv/d10_scalar_s0.pt/memoryless.pt")
    sd = torch.load(ck, map_location="cpu")
    m = CalibScorer(sd.get("d_kv",128), sd["L"], sd["H"], n_slots=sd.get("slots",8),
                    d_m=sd.get("dim",128), mode="memoryless", arch=sd["arch"])
    m.load_state_dict(sd["state"]); m.eval()
    alpha = float(m.alpha)
    print(f"ckpt={sd['arch']}  alpha={alpha:.4f}")

    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "scratch_ctrl_traces_v2_10/doc*.pt")))[:3]:
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            g, t = float(ch["gsig"]), float(ch["thres"])
            for l, pl in enumerate(ch["layers"]):
                n = pl["n_near"]
                s0 = pl["s0"][:, :n].float()
                mu = pl["mu_h"].float(); sg = pl["sig_h"].float()
                mg = (s0 - t) / g
                with torch.no_grad():
                    ds = m.delta(None, m.read(m.init_state(l), None), s0,
                                 margin=mg, stats=(mu, sg, torch.tensor(g)))
                for h in range(s0.shape[0]):
                    j = int(torch.argmax(s0[h]))
                    smax = float(s0[h, j])
                    rows.append((l, h, smax, t, float(sg[h]), float(ds[h, j]),
                                 float(mu[h]), float(s0[h].mean())))
    A = np.array([r[2:] for r in rows])            # smax, tau, sig, dsmax, mu, smean
    smax, tau, sig, dsm = A[:,0], A[:,1], A[:,2], A[:,3]
    gap = tau - smax
    starved = gap > 0
    print(f"\n{len(A)} 个 (chunk,层,头)；其中候选池内最高分仍低于 τ 的 = "
          f"{starved.sum()} ({starved.mean()*100:.1f}%)  ← trace 上的「饿死」代理")

    for nm, m_ in (("饿死头", starved), ("非饿死头", ~starved)):
        if m_.sum() == 0: continue
        print(f"\n【{nm}】n={m_.sum()}")
        print(f"  σ_h            中位 {np.median(sig[m_]):9.4f}   "
              f"p10 {np.percentile(sig[m_],10):9.4f}  p90 {np.percentile(sig[m_],90):9.4f}")
        if nm == "饿死头":
            U = alpha*sig[m_]/np.maximum(gap[m_], 1e-12)
            C = dsm[m_]/np.maximum(gap[m_], 1e-12)
            print(f"  缺口 τ−s_max   中位 {np.median(gap[m_]):9.4f}")
            print(f"  **物理上限 U** 中位 {np.median(U):9.4f}   "
                  f"≥1 的比例 {(U>=1).mean()*100:5.1f}%   ← 假说2：U≪1 则抬不动")
            print(f"  **实际 C**     中位 {np.median(C):9.4f}   "
                  f"≥1 的比例 {(C>=1).mean()*100:5.1f}%")
            print(f"  **Δs(s_max) 的符号**：正 {(dsm[m_]>0).mean()*100:5.1f}%  "
                  f"负 {(dsm[m_]<0).mean()*100:5.1f}%   ← 假说3：多数为负则方向反了")
            print(f"  Δs(s_max) 中位 {np.median(dsm[m_]):+.6f}")
    print(f"\n非饿死头的 Δs 符号：正 {(dsm[~starved]>0).mean()*100:.1f}% / "
          f"负 {(dsm[~starved]<0).mean()*100:.1f}%")


if __name__ == "__main__":
    raise SystemExit(main())
