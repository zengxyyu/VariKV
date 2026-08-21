"""从已有的 set_marginal trace 直接构造「固定预算转移优势」，并检验它是否可学。

数学（外部评审提的 counterfactual allocation advantage，这里落到可算的形式）：

    A_{i←j}^{(k)} = J(b⁰ + k·e_i − k·e_j) − J(b⁰)
                  ≈ Σ_{头 i 中最好的 k 个被驱逐者} U  −  Σ_{头 j 中最差的 k 个保留者} U

其中 U 是**条件于真实存活集合**的边际效用（`--utility set_marginal`），
所以「加回来」与「拿掉」的价值都用同一把尺子量。预算守恒**精确成立**。

关键结构：`A_{i←j} = g⁺_i − g⁻_j`，**在 i 与 j 上可分**。于是
  * 最佳受主 = argmax g⁺；最佳施主 = argmin g⁻；
  * **有益的转移存在 ⟺ max g⁺ > min g⁻** —— 这就是 no-op 判据，
    不是外加的启发式。
  * 若 g⁺_h ≈ g⁻_h ≡ u_h（前沿边际效用），则退化成 KKT 条件
    「最优时各头边际效用相等」，而 CHead 的 c_h 只需与 u_h 同序。

本脚本零 GPU，回答三个问题：
  ① u_h 有没有信号（跨头离散度 vs 跨 chunk 噪声）；
  ② no-op 判据 max g⁺ > min g⁻ 有多常触发；
  ③ **现在训好的 chd10 的 c_h 与 u_h 相关吗** —— 若不相关，
     说明现用教师教的根本不是这个量，那是直接证据。
"""
import os, sys, glob, argparse, numpy as np, torch
from scipy import stats as st
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.calib_scorer import CalibScorer                      # noqa: E402


def load_ctrl(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    m = CalibScorer(sd.get("d_kv", 128), sd["L"], sd["H"],
                    n_slots=sd.get("slots", 8), d_m=sd.get("dim", 128),
                    mode="memoryless", arch=sd["arch"],
                    scale=sd.get("scale", "head"),
                    alpha_max=sd.get("alpha_max", 1.0))
    m.load_state_dict(sd["state"])
    return m.eval()


def frontier(pl, k=1):
    """→ (g_plus[H], g_minus[H])，单位与 U 一致。缺侧返回 nan。"""
    s0 = pl["s0"].float(); U = pl["U"].float(); ret = pl["ret"]
    H = s0.shape[0]
    gp = np.full(H, np.nan); gm = np.full(H, np.nan)
    for h in range(H):
        ev = (~ret[h]).nonzero(as_tuple=True)[0]        # 被驱逐的候选
        rt = ret[h].nonzero(as_tuple=True)[0]           # 保留的候选
        if len(ev) >= k:                                 # 最好的 k 个被驱逐者
            top = ev[torch.argsort(s0[h][ev], descending=True)[:k]]
            gp[h] = float(U[h][top].sum())
        if len(rt) >= k:                                 # 最差的 k 个保留者
            bot = rt[torch.argsort(s0[h][rt])[:k]]
            gm[h] = float(U[h][bot].sum())
    return gp, gm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="scratch_ctrl_traces_sm_cont")
    ap.add_argument("--n_doc", type=int, default=10)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--ctrl", default="varikv/chead10_s0.pt/memoryless.pt")
    a = ap.parse_args()

    GP, GM, CH, KEY = [], [], [], []
    ctrl = load_ctrl(os.path.join(ROOT, a.ctrl))
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(ROOT, a.traces, "doc*.pt")))[:a.n_doc]:
            d = torch.load(f, map_location="cpu", weights_only=False)
            for ci, ch in enumerate(d["chunks"]):
                g, t = float(ch["gsig"]), float(ch["thres"])
                for l, pl in enumerate(ch["layers"]):
                    gp, gm = frontier(pl, a.k)
                    s0 = pl["s0"].float()
                    mu, sh = pl["mu_h"].float(), pl["sig_h"].float()
                    ds = ctrl.delta(None, ctrl.read((l,), None), s0,
                                    margin=(s0 - t) / g, stats=(mu, sh, g))
                    c = (ds[:, 0] / g).numpy()
                    for h in range(len(gp)):
                        GP.append(gp[h]); GM.append(gm[h]); CH.append(c[h])
                        KEY.append((d["doc"], ci, l, h))
    GP, GM, CH = map(np.array, (GP, GM, CH))
    ok = ~np.isnan(GP) & ~np.isnan(GM)
    print(f"样本 {len(GP)} 个 (doc,chunk,层,头)，两侧都有的 {ok.sum()}")

    u = np.where(ok, (GP + GM) / 2, np.nan)
    print(f"\n① 前沿边际效用 u_h（两侧均值）")
    print(f"   跨样本 sd {np.nanstd(u):.5f}   中位 {np.nanmedian(u):.5f}   "
          f"负值占 {np.nanmean(u[ok] < 0):.1%}")
    print(f"   g⁺（加回来的价值）中位 {np.nanmedian(GP):.5f}")
    print(f"   g⁻（拿掉的代价）  中位 {np.nanmedian(GM):.5f}")
    print(f"   ⇒ g⁻ > g⁺ 的比例 {np.nanmean(GM[ok] > GP[ok]):.1%}"
          f"（保留者更值钱 ⇒ 符合预期）")

    print(f"\n② no-op 判据：逐 chunk 看 max g⁺ 是否 > min g⁻")
    import collections
    per = collections.defaultdict(lambda: ([], []))
    for (doc, ci, l, h), p, m_ in zip(KEY, GP, GM):
        if not np.isnan(p): per[(doc, ci)][0].append(p)
        if not np.isnan(m_): per[(doc, ci)][1].append(m_)
    fire = [max(p) > min(mm) for p, mm in per.values() if p and mm]
    print(f"   {len(fire)} 个 chunk 中，**存在有益转移的占 {np.mean(fire):.1%}**")
    print(f"   ⇒ 另外 {1-np.mean(fire):.1%} 的 chunk 上，最优动作是**什么都不做**")

    print(f"\n③ 现用 chd10 的 c_h 与 u_h 的关系（**关键检验**）")
    m2 = ok & ~np.isnan(CH)
    print(f"   Pearson  {st.pearsonr(CH[m2], u[m2])[0]:+.4f}  "
          f"p={st.pearsonr(CH[m2], u[m2])[1]:.2e}")
    print(f"   Spearman {st.spearmanr(CH[m2], u[m2])[0]:+.4f}  "
          f"p={st.spearmanr(CH[m2], u[m2])[1]:.2e}")
    print(f"   对照 c_h vs g⁺  Spearman {st.spearmanr(CH[m2], GP[m2])[0]:+.4f}")
    print(f"   对照 c_h vs g⁻  Spearman {st.spearmanr(CH[m2], GM[m2])[0]:+.4f}")
