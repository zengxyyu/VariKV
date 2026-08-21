"""各臂的**每一个输入特征**在 ρ=0.1 与 ρ=0.3 之间移动了多少。

为什么这能解释「scalar 跨 ratio 稳、chead 崩」：
两条臂的输入清单不同，而其中**只有含 τ 的那个分量随 ratio 平移**：

    scalar : z = (s⁰−μ_h)/σ_h   [与 τ 无关]
             mg = (s⁰−τ)/σ_g    [**随 ρ 平移**]
             rs = log(σ_h/σ_g)  [与 τ 无关]
    chead  : mgm = (μ_h−τ)/σ_g  [**随 ρ 平移**，且是**逐头聚合量**]
             rs                 [与 τ 无关]

关键差别有两层：
 ① **占比**：chead 的 2 个输入里有 1 个随 ρ 走；scalar 的 3 个里也有 1 个，
    但另外两个（尤其逐 token 的 z）承载了主要信息；
 ② **集中度**：`mg` 是**逐 token** 的，天然跨越很宽的范围，训练时就见过
    远离阈值的取值；`mgm` 是**逐头均值**，分布窄得多，ρ 一变就整体挪出训练区。

度量：每个特征在两个 ratio 下分布的**重叠系数**（1=完全重合，0=不相交）与 KS。
零 GPU。
"""
import glob, numpy as np, torch
from scipy import stats as st


def collect(tr, n=10):
    Z, MG, RS, MGM = [], [], [], []
    for f in sorted(glob.glob(f"{tr}/doc*.pt"))[:n]:
        d = torch.load(f, map_location="cpu", weights_only=False)
        for ch in d["chunks"]:
            g, t = float(ch["gsig"]), float(ch["thres"])
            for pl in ch["layers"]:
                s0 = pl["s0"].float(); mu = pl["mu_h"].float(); sh = pl["sig_h"].float()
                Z.append(((s0 - mu[:, None]) / sh[:, None].clamp_min(1e-6)).numpy().ravel())
                MG.append(((s0 - t) / g).numpy().ravel())
                RS.append(np.log((sh / g).clamp_min(1e-9).numpy()))
                MGM.append(((mu - t) / g).numpy())
    return (np.concatenate(Z), np.concatenate(MG),
            np.concatenate(RS), np.concatenate(MGM))


def overlap(a, b, nb=300):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    e = np.linspace(lo, hi, nb)
    ha, _ = np.histogram(a, bins=e, density=True)
    hb, _ = np.histogram(b, bins=e, density=True)
    return float(np.minimum(ha, hb).sum() * (e[1] - e[0]))


if __name__ == "__main__":
    A = collect("scratch_ctrl_traces_v2_10")
    B = collect("scratch_ctrl_traces_r03")
    names = [("z   (scalar，逐 token)", 0, "与 τ 无关"),
             ("mg  (scalar，逐 token)", 1, "**随 ρ 平移**"),
             ("rs  (两臂共有，逐头)", 2, "与 τ 无关"),
             ("mgm (chead，逐头聚合)", 3, "**随 ρ 平移**")]
    print(f"{'特征':<24}{'重叠系数':>10}{'KS':>8}{'中位移动':>10}  性质")
    ov = {}
    for nm, i, note in names:
        a, b = A[i], B[i]
        # 子采样，逐 token 特征有 2000 万点
        if a.size > 400000:
            rg = np.random.default_rng(0)
            a = a[rg.choice(a.size, 400000, replace=False)]
            b = b[rg.choice(b.size, 400000, replace=False)]
        o = overlap(a, b); k = st.ks_2samp(a, b).statistic
        d = float(np.median(b) - np.median(a)); ov[nm] = o
        print(f"{nm:<24}{o:>10.3f}{k:>8.3f}{d:>+10.3f}  {note}")
    print(f"\n**结论比值**：chead 的 `mgm` 重叠 {ov['mgm (chead，逐头聚合)']:.3f} "
          f"vs scalar 的 `mg` 重叠 {ov['mg  (scalar，逐 token)']:.3f}"
          f"  ⇒ 相差 {ov['mg  (scalar，逐 token)']/max(ov['mgm (chead，逐头聚合)'],1e-9):.1f} 倍")
