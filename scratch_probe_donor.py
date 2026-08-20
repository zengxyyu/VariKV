#!/usr/bin/env python3
"""donor 审计：层带臂之间「被扣预算的头」是否逐位相同 —— 零 GPU。

**为什么必须有**（外部复核 2026-08-20 提出，采纳）：固定总预算下
「给某些头 +1」必然从别的头扣，所以每个 floorcov/band 臂都是
**预算转移**而非纯增加。若 donor 随 receiver 变化，则
「唯一变量是层带」不成立，+25.40 vs −0.80 就混进了 donor 侧效应。

读代码看 `room = (b0 - bmin).clamp(min=0)` 只依赖 b0 与 bmin、与 `pick` 无关，
**但读代码不算测量**。本探针在真实 trace 上直接算出两臂的完整 Δb 向量，
比对其**负分量**（donor 侧）。
"""
import glob, os, sys
import numpy as np, torch
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.quota_project import project_quota                    # noqa: E402

H = 4


def arm(b0, delta, n, L, Hh, sc, band, N, order="band"):
    os.environ.update({"VARIKV_QUOTA_FLOOR": "1", "VARIKV_COV_ORDER": order,
                       "VARIKV_COV_BAND": band, "VARIKV_COV_N": str(N)})
    return project_quota(b0.clone(), delta, n, "floorcov", L, Hh, sc=sc)


def main():
    rows = []
    skipped = 0
    for fp in sorted(glob.glob(f"{ROOT}/scratch_ctrl_traces_v2_10/doc*.pt"))[:5]:
        d = torch.load(fp, map_location="cpu")
        for ch in d["chunks"]:
            t = float(ch["thres"])
            sc = torch.cat([pl["s0"][:, :pl["n_near"]].float() for pl in ch["layers"]], 0)
            G, npt = sc.shape
            L = G // H
            s0f = sc.reshape(-1)
            B = int((s0f > t).sum())
            if B < G or B >= len(s0f) - G:
                skipped += 1
                continue
            vb = torch.zeros(len(s0f), dtype=torch.bool)
            vb[torch.topk(s0f, B).indices] = True
            b0 = vb.reshape(G, npt).sum(-1).float()
            dl = torch.zeros(G)
            qa = arm(b0, dl, npt, L, H, sc, "0-1", 8)
            qb = arm(b0, dl, npt, L, H, sc, "2-5", 8)
            da = qa.float() - b0
            db = qb.float() - b0
            don_a = da.clamp(max=0)                 # donor 侧（负分量）
            don_b = db.clamp(max=0)
            rows.append((
                float((don_a - don_b).abs().sum()),          # donor 逐位差 L1
                float(-don_a.sum()), float(-don_b.sum()),     # 各自扣掉的总量
                int((don_a < 0).sum()), int((don_b < 0).sum()),
                int(((don_a < 0) ^ (don_b < 0)).sum()),       # donor 集合的对称差
                float(da.clamp(min=0).sum()), float(db.clamp(min=0).sum()),
            ))
    A = np.array(rows)
    print(f"{len(rows)} 个 chunk（跳过 {skipped} 个不满足 B 范围的）")
    print(f"\n【donor 侧逐位比对】L0-1 臂 vs L2-5 臂")
    print(f"  donor 向量 L1 差    中位 {np.median(A[:,0]):.6f}   最大 {A[:,0].max():.6f}"
          f"   完全相同的 chunk 数 **{int((A[:,0]==0).sum())}/{len(A)}**")
    print(f"  donor 集合对称差    中位 {np.median(A[:,5]):.1f}   最大 {A[:,5].max():.0f}")
    print(f"  扣掉的总量          L0-1 {np.mean(A[:,1]):.3f}   L2-5 {np.mean(A[:,2]):.3f}")
    print(f"  donor 头数          L0-1 {np.mean(A[:,3]):.1f}    L2-5 {np.mean(A[:,4]):.1f}")
    print(f"  receiver 抬升总量   L0-1 {np.mean(A[:,6]):.3f}   L2-5 {np.mean(A[:,7]):.3f}")
    same = int((A[:, 0] == 0).sum())
    print(f"\n判词：donor 侧逐位相同的 chunk 占 **{100*same/len(A):.1f}%**"
          f" ⇒ {'**donor 混淆不存在，两臂只差 receiver 层带**' if same == len(A) else '**donor 随 receiver 变化，需在文档中标注为混淆**'}")
    return 0 if same == len(A) else 1


if __name__ == "__main__":
    raise SystemExit(main())
