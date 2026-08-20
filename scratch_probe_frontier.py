#!/usr/bin/env python3
"""阈值前沿的配额响应：一阶密度 + 饿死头的死区 —— 零 GPU。

**推导**（外部复核给出一阶项，死区一段是本文补的）：
    b_h(δ_h) = #{i : s_{h,i} + δ_h > τ'}
             = n_h (1 − F_h(τ' − δ_h))
  ⇒ 小扰动下  Δb_h ≈ ρ_h(τ)·(δ_h − Δτ),   ρ_h(τ) := n_h f_h(τ)
  预算守恒 Σ_h Δb_h = 0 ⇒ **Δτ = Σ_h ρ_h δ_h / Σ_h ρ_h**

**但这个一阶式对饿死头无效。** 饿死头满足 s_max,h < τ ⇒ 阈值处密度
`ρ_h(τ) = 0` ⇒ 一阶响应恒为 0。真实响应是**死区**：
    δ_h < τ − s_max,h  ⇒  Δb_h = 0（严格）
    δ_h ≥ τ − s_max,h  ⇒  开始跳变
所以「要多大的动作」由**死区宽度**而非密度决定，这正是 `C*` 分析在测的东西。

本探针同时给出两个量，并按层聚合，检验它们能否解释层剖面
（L1 四头 +24.80★ vs L0 −0.80 ns vs L2 +0.00 ns）。
"""
import glob, os, sys
import numpy as np, torch

ROOT = os.path.abspath(os.path.dirname(__file__))
H = 4


def main():
    eps = float(os.environ.get("EPS", "0.1"))          # 密度窗口，单位 σ_g
    rows, skipped = [], 0
    for fp in sorted(glob.glob(f"{ROOT}/scratch_ctrl_traces_v2_10/doc*.pt")):
        d = torch.load(fp, map_location="cpu")
        for ch in d["chunks"]:
            t = float(ch["thres"]); g = float(ch["gsig"])
            sc = torch.cat([pl["s0"][:, :pl["n_near"]].float() for pl in ch["layers"]], 0)
            G, npt = sc.shape
            s0f = sc.reshape(-1)
            B = int((s0f > t).sum())
            if B < G or B >= len(s0f) - G:
                skipped += 1; continue
            vb = torch.zeros(len(s0f), dtype=torch.bool)
            vb[torch.topk(s0f, B).indices] = True
            b0 = vb.reshape(G, npt).sum(-1)
            smax = sc.max(-1).values
            # 阈值局部密度（按 σ_g 归一化的窗口内计数）
            rho = ((sc - t).abs() < eps * g).sum(-1).float()
            gap = (t - smax) / g                        # 死区宽度（σ_g 单位）；<0 表示已越阈
            for h in range(G):
                rows.append((h // H, int(b0[h] == 0), float(rho[h]), float(gap[h])))
    A = np.array(rows)
    lay, starved, rho, gap = A[:, 0], A[:, 1].astype(bool), A[:, 2], A[:, 3]
    print(f"{len(rows)} 个 (chunk,层,头)，跳过 {skipped} 个 chunk；密度窗口 ±{eps} σ_g")
    print(f"\n【一阶式对饿死头是否失效】")
    print(f"  饿死头（b⁰=0）占 {100*starved.mean():.1f}%")
    print(f"  **饿死头的阈值局部密度 ρ_h(τ)：中位 {np.median(rho[starved]):.1f}"
          f"，为 0 的比例 {100*(rho[starved]==0).mean():.1f}%**")
    print(f"  非饿死头 ρ_h(τ)：中位 {np.median(rho[~starved]):.1f}")
    print(f"  ⇒ {'**一阶式对饿死头失效**（ρ≈0 ⇒ 线性响应恒 0），必须用死区刻画'
             if np.median(rho[starved]) < np.median(rho[~starved]) else '一阶式对两者都可用'}")
    print(f"\n【死区宽度 τ − s_max,h（σ_g 单位），只看饿死头，按层带】")
    print(f"{'层带':<10}{'头次':>7}{'死区中位':>10}{'p90':>9}{'ρ 中位':>9}")
    for lo, hi, nm in [(0, 0, "L0"), (1, 1, "L1"), (2, 2, "L2"),
                       (3, 5, "L3-5"), (6, 11, "L6-11"), (12, 19, "L12-19"), (20, 27, "L20-27")]:
        m = starved & (lay >= lo) & (lay <= hi)
        if m.sum() == 0:
            continue
        print(f"{nm:<10}{int(m.sum()):>7}{np.median(gap[m]):>10.3f}"
              f"{np.percentile(gap[m], 90):>9.3f}{np.median(rho[m]):>9.1f}")
    print(f"\n  参照：`C*` 中位 = 1.163 σ_g（使地板配额可达的统一界）")
    print(f"  参照：当前 α·σ_h 中位 = 0.137 σ_g；新方法 α·σ_g = 1.25 σ_g")


if __name__ == "__main__":
    raise SystemExit(main())
