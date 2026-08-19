"""任意增益 g 下的头内单调性证书 —— 一次算完全部 g，零 GPU。

原探针只报 `1 + a*sech^2(phi)*phi'` 即 g=+1 的余量。一般情形是

    ds'/ds = 1 + g * d'(z),      d'(z) = alpha * sech^2(phi(z)) * phi'(z)

所以**只要报出 d' 在全部实测状态 x 全部网格点上的 min 与 max，
就一次性给出所有 g 的单调区间**：

    g > 0 单调  <=>  g * min(d') > -1  <=>  g < -1/min(d')   (当 min d' < 0)
    g < 0 单调  <=>  g * max(d') > -1  <=>  |g| < 1/max(d')  (当 max d' > 0)

这条很要紧：g=+1 单调（1+min d' > 0）**不蕴含** g=-1 单调（要 1-max d' > 0）。
两者约束的是 d' 的**相反两端**。
"""
import glob, os, sys
import numpy as np, torch
ROOT = "/home/ubuntu/zxy/vlm-memory"
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
sys.path.insert(0, ROOT)
from attention.calib_scorer import CalibScorer
import importlib.util
spec = importlib.util.spec_from_file_location("pm", os.path.join(ROOT, "scratch_probe_monotone.py"))
pm = importlib.util.module_from_spec(spec); spec.loader.exec_module(pm)

CK = sys.argv[1] if len(sys.argv) > 1 else "varikv/d10_scalar_s0.pt/memoryless.pt"
m, arch = pm.load(os.path.join(ROOT, CK))
alpha = float(m.alpha)
st, zlo, zhi = pm.states("scratch_ctrl_traces_v2_10", 2)
lo, hi = zlo - 2.0, zhi + 2.0
z = torch.linspace(lo, hi, 4001, dtype=torch.float32)
feats = CalibScorer.SCALAR_FEATS[arch]; use_emb = arch not in CalibScorer.NO_EMB
dmin, dmax, at_min, at_max = np.inf, -np.inf, None, None
for (l, h, A, B) in st:
    zz = z.clone().requires_grad_(True)
    col = {"z": zz, "mg": A * zz + B,
           "rs": torch.full_like(zz, float(np.log(max(A, 1e-12))))}
    parts = [col[k][:, None] for k in feats]
    if use_emb:
        parts.append(m.emb[l, h][None, :].expand(len(zz), -1))
    phi = m.head(torch.cat(parts, -1)).squeeze(-1)
    dphi, = torch.autograd.grad(phi.sum(), zz)
    dp = (alpha * (1.0 - torch.tanh(phi) ** 2) * dphi).detach()
    if float(dp.min()) < dmin: dmin, at_min = float(dp.min()), (l, h, float(z[int(dp.argmin())]))
    if float(dp.max()) > dmax: dmax, at_max = float(dp.max()), (l, h, float(z[int(dp.argmax())]))
print(f"{CK}  arch={arch}  alpha={alpha:.4f}")
print(f"  {len(st)} 组实测 (chunk,层,头) x 4001 网格点，z in [{lo:.1f},{hi:.1f}]")
print(f"  min d'(z) = {dmin:+.6f}  at 层{at_min[0]} 头{at_min[1]} z={at_min[2]:+.2f}")
print(f"  max d'(z) = {dmax:+.6f}  at 层{at_max[0]} 头{at_max[1]} z={at_max[2]:+.2f}")
print()
print(f"  {'g':>6}{'min ds/ds':>14}   判定")
for g in (-4.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
    mg = 1.0 + (g * dmin if g > 0 else g * dmax)
    print(f"  {g:>6.1f}{mg:>14.6f}   {'单调' if mg > 0 else '**非单调**'}")
gpos = (-1.0 / dmin) if dmin < 0 else float("inf")
gneg = (-1.0 / dmax) if dmax > 0 else -float("inf")
print(f"\n  ⇒ 单调区间 g in ({gneg:.4f}, {gpos:.4f})")
