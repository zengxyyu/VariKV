"""解析 scratch_probe_damage.pt（NEXT_STEPS.md v5 §4 P0-B 的报告）。

三条报告纪律（v5）：
  1. **相关性必须先聚合到 token 级**。一个 query token 的 B 被 28 层 × 28 q-head = 784 个点
     共享，直接 flatten 是伪重复，有效样本量被夸大近三个数量级。
  2. **首要看 G_proj / G_layer，不是 value 空间的 G**。跨 head/layer 唯一可比的是经 W_O 投影后的量。
  3. **不要只报均值**：M 可能极度重尾，必须报 median/P90/P95/P99/max，以及条件量 E[·|B top10%]。
"""
import sys

import numpy as np
import torch


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def qs(x, name, unit=""):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    print(f"  {name:<22} n={x.size:<8d} mean={x.mean():9.5f}  median={np.median(x):9.5f}  "
          f"P90={np.quantile(x,.9):9.5f}  P95={np.quantile(x,.95):9.5f}  "
          f"P99={np.quantile(x,.99):9.5f}  max={x.max():9.5f}{unit}")


def main(path="scratch_probe_damage.pt"):
    recs = torch.load(path, map_location="cpu", weights_only=False)
    nq = sum(len(r["queries"]) for r in recs)
    print(f"载入 {len(recs)} 样本 / {nq} 个 query，上下文 "
          f"{min(r['n_ctx'] for r in recs)}–{max(r['n_ctx'] for r in recs)} tok")

    # ---------- 逐 query 聚合（每个 query token 一个观测）----------
    rows = []          # (B, Gproj_max, Gproj_sum, Glayer_max, G_max, M_max, M_mean, C_max)
    perlayer = {}      # l -> list of (Gproj_mean_over_heads, M_mean, Glayer)
    allM, allC, allG, allGp = [], [], [], []

    for r in recs:
        for q in r["queries"]:
            B = q["B"].numpy()                       # [T]
            T = B.shape[0]
            gp = np.zeros((0, T)); gl = np.zeros((0, T))
            Ms = np.zeros((0, T)); Cs = np.zeros((0, T)); Gs = np.zeros((0, T))
            for l, d in q["layers"].items():
                Gp = d["Gproj"].numpy()              # [HQ,T]
                gp = np.concatenate([gp, Gp])
                gl = np.concatenate([gl, d["Glayer"].numpy().reshape(1, T)])
                Ms = np.concatenate([Ms, d["M"].numpy()])
                Cs = np.concatenate([Cs, d["C"].numpy()])
                Gs = np.concatenate([Gs, d["G"].numpy()])
                perlayer.setdefault(l, []).append(
                    (float(Gp.mean()), float(d["M"].numpy().mean()),
                     float(d["Glayer"].numpy().mean())))
            allM.append(Ms.ravel()); allC.append(Cs.ravel())
            allG.append(Gs.ravel()); allGp.append(gp.ravel())
            # 逐 token 一个观测；is_last 标记问题的最后一个 token（答案开始处）
            for t in range(T):
                # M_max 是 784 个点的最大值，几乎恒为 1（饱和），不能用来分箱/相关。
                # 用 mean 与 P90，以及 G_proj 的 top-10 均值。
                rows.append((B[t], gp[:, t].max(), gp[:, t].sum(),
                             gl[:, t].max(), Gs[:, t].mean(),
                             Ms[:, t].mean(), np.quantile(Ms[:, t], .9),
                             Cs[:, t].mean(),
                             np.sort(gp[:, t])[-10:].mean(), gl[:, t].mean(),
                             1.0 if t == T - 1 else 0.0))

    A = np.array(rows)
    (B_, Gp_max, Gp_sum, Gl_max, G_mean, M_mean, M_p90, C_mean,
     Gp_top10, Gl_mean, is_last) = A.T

    print("\n" + "=" * 96)
    print("一、遗漏质量 M 的分布（逐 (层, q-head, token)，全部点）")
    qs(np.concatenate(allM), "M = D_E/(D_R+D_E)")
    qs(np.concatenate(allC), "C = ‖o_E−o_R‖")
    qs(np.concatenate(allG), "G = M·C (value 空间)")
    qs(np.concatenate(allGp), "G_proj = ‖W_O·Δo‖")

    print("\n二、条件分布：驱逐真正伤到行为的 query 上，M 是否更大？")
    hi = B_ >= np.quantile(B_, 0.9)
    lo = B_ <= np.quantile(B_, 0.5)
    for nm, v in (("M_mean", M_mean), ("M_p90", M_p90), ("C_mean", C_mean),
                  ("G_mean(value)", G_mean), ("G_proj_max", Gp_max),
                  ("G_proj_top10", Gp_top10), ("G_proj_sum", Gp_sum),
                  ("G_layer_max", Gl_max), ("G_layer_mean", Gl_mean)):
        print(f"  {nm:<14} B∈top10% 均值 {v[hi].mean():10.5f}   "
              f"B∈bottom50% 均值 {v[lo].mean():10.5f}   "
              f"比值 {v[hi].mean()/max(v[lo].mean(),1e-12):7.2f}×")

    print("\n三、token 级相关（每个 query token 一个观测，**不 flatten**）")
    print(f"  观测数 n = {len(B_)}")
    for nm, v in (("G_mean (value 空间)", G_mean), ("G_proj_max", Gp_max),
                  ("G_proj_top10", Gp_top10), ("G_proj_sum", Gp_sum),
                  ("G_layer_max", Gl_max), ("G_layer_mean", Gl_mean),
                  ("M_mean", M_mean), ("M_p90", M_p90), ("C_mean", C_mean)):
        print(f"  spearman(B, {nm:<20}) = {spearman(B_, v):+.4f}")

    print("\n四、逐层（G_proj 均值降序，前 10 与后 5）")
    agg = {l: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])),
               float(np.mean([x[2] for x in v]))) for l, v in perlayer.items()}
    order = sorted(agg, key=lambda l: -agg[l][0])
    print(f"  {'layer':>5} {'G_proj均值':>12} {'M均值':>10} {'G_layer均值':>12}")
    for l in order[:10] + ["…"] + order[-5:]:
        if l == "…":
            print("     …"); continue
        a = agg[l]
        print(f"  {l:>5} {a[0]:>12.5f} {a[1]:>10.5f} {a[2]:>12.5f}")

    print("\n五、跨头相消：Σ_h‖W_O Δo_h‖  vs  ‖Σ_h W_O Δo_h‖")
    ratios = []
    for r in recs:
        for q in r["queries"]:
            for l, d in q["layers"].items():
                s = d["Gproj"].numpy().sum(0)          # [T]  各头范数之和
                j = d["Glayer"].numpy()                # [T]  联合后的范数
                ratios.append(j / np.maximum(s, 1e-12))
    ratios = np.concatenate(ratios)
    qs(ratios, "‖Σ‖ / Σ‖·‖")
    print("  （≪1 ⇒ 跨头显著相消 ⇒ 逐头独立修正在对抗相消，见 v5 §1.6）")

    print("\n六、三张联合图的数值版：按 M 分箱看 C 与 B")
    for bname, bv in (("M_mean", M_mean), ("G_proj_top10", Gp_top10)):
        edges = np.quantile(bv, [0, .2, .4, .6, .8, 1.0])
        print(f"\n  按 {bname} 五分箱：")
        print(f"  {'分箱':>24} {'n':>5} {'C_mean':>10} {'G_proj_top10':>13} "
              f"{'G_layer_mean':>13} {'B均值':>10} {'B中位':>10}")
        for i in range(5):
            sel = (bv >= edges[i]) & (bv <= edges[i + 1])
            if sel.sum() == 0:
                continue
            print(f"  [{edges[i]:9.4f},{edges[i+1]:9.4f}] {sel.sum():>5} "
                  f"{C_mean[sel].mean():>10.4f} {Gp_top10[sel].mean():>13.4f} "
                  f"{Gl_mean[sel].mean():>13.4f} {B_[sel].mean():>10.4f} "
                  f"{np.median(B_[sel]):>10.4f}")
    # ---------- 七、只看每个问题的最后一个 token ----------
    # 问题前缀 token（"Q: What is the value of key …"）的预测几乎不依赖上下文，
    # 混进来会把信号稀释。真正需要检索的是最后一个 token（答案开始处）。
    sel = is_last > 0.5
    print(f"\n七、只看问题末 token（答案开始处），n = {int(sel.sum())}")
    qs(B_[sel], "B (末 token)")
    print("  token 级 spearman：")
    for nm, v in (("G_proj_max", Gp_max), ("G_proj_top10", Gp_top10),
                  ("G_proj_sum", Gp_sum), ("G_layer_max", Gl_max),
                  ("G_layer_mean", Gl_mean), ("M_mean", M_mean),
                  ("M_p90", M_p90), ("C_mean", C_mean)):
        print(f"    spearman(B, {nm:<16}) = {spearman(B_[sel], v[sel]):+.4f}")
    hi2 = sel & (B_ >= np.quantile(B_[sel], 0.75))
    lo2 = sel & (B_ <= np.quantile(B_[sel], 0.5))
    print("  条件对比（末 token 内部，B 上四分位 vs 下半）：")
    for nm, v in (("M_mean", M_mean), ("C_mean", C_mean),
                  ("G_proj_top10", Gp_top10), ("G_layer_mean", Gl_mean)):
        print(f"    {nm:<14} 高B {v[hi2].mean():10.5f}   低B {v[lo2].mean():10.5f}   "
              f"比值 {v[hi2].mean()/max(v[lo2].mean(),1e-12):6.2f}×")
    print("=" * 96)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scratch_probe_damage.pt")
