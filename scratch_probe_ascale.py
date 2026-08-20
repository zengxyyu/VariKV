"""A_h = sig_h/sig_g 的分布，以及「同 α 下 σ_g 是否也放大了平均动作」这个混淆。

σ_g 臂在同一个 α 下，每个头的动作半径从 α·σ_h 变成 α·σ_g，比值 1/A_h。
若 E[A_h] < 1 则**平均动作也变大了** —— 那么 +13.53 就不是纯重加权。
本脚本只读 trace（零 GPU），把这个量算出来。
"""
import torch, glob, numpy as np

A, SH, SG = [], [], []
for f in sorted(glob.glob("scratch_ctrl_traces_v2/*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    for c in d["chunks"]:
        g = float(c["gsig"])
        for lay in c["layers"]:
            sh = lay["sig_h"].float().numpy()
            A.append(sh / g); SH.append(sh); SG.append(np.full_like(sh, g))
A = np.concatenate(A); SH = np.concatenate(SH); SG = np.concatenate(SG)
q = lambda x, p: float(np.percentile(x, p))

print(f"样本 = {A.size} 个 (doc,chunk,层,头)")
print(f"\nA_h = σ_h/σ_g")
print(f"  min {A.min():.4f}  p10 {q(A,10):.4f}  中位 {q(A,50):.4f}  "
      f"均值 {A.mean():.4f}  p90 {q(A,90):.4f}  max {A.max():.4f}")
print(f"  跨头动态范围 max/min = {A.max()/A.min():.0f}×")

print(f"\n=== 混淆检验：同 α 下两种参数化的**平均动作半径**")
print(f"  σ_h 臂  E[α·σ_h] = α × {SH.mean():.6f}")
print(f"  σ_g 臂  E[α·σ_g] = α × {SG.mean():.6f}")
r = SG.mean() / SH.mean()
print(f"  **比值 = {r:.3f}×**  ⇒ σ_g 在同 α 下平均动作是 σ_h 的 {r:.2f} 倍")
print(f"  （注意 E[σ_g]/E[σ_h] = 1/E[A_h] 只在 σ_g 为常数时成立；这里逐 chunk 变，故直接取均值）")

print(f"\n=== 若要给 σ_h 臂做**等平均幅度**对照，需要的 α")
for a in (0.999,):
    print(f"  σ_g 臂 α={a} ⇒ 平均半径 {a*SG.mean():.6f}")
    print(f"  σ_h 臂 要达到同一平均半径需 **α = {a*r:.3f}**")

print(f"\n=== 分位数上的重加权幅度（每个头自己的半径变化倍数 1/A_h）")
inv = 1.0 / A
print(f"  1/A_h: p10 {q(inv,10):.2f}×  中位 {q(inv,50):.2f}×  p90 {q(inv,90):.2f}×  max {inv.max():.0f}×")
print(f"  被**缩小**的头（A_h>1，即 σ_g 臂半径更小）占 {100*(A>1).mean():.2f}%")
print(f"  被放大 >10× 的头占 {100*(inv>10).mean():.2f}%")
