"""逐种子的配额去向：方差是「同一个解上的噪声」还是「不同的解」？

`chd0`（无头嵌入）三种子分数 +25.00/+21.40/+16.20，跨度 8.8 分；
`chd10`（有头嵌入）+24.40/+23.80/+24.40，跨度 0.6 分。
若前者的方差来自**不同种子把预算搬去不同的层**，那是「多解」；
若三个种子搬去同样的层、只是量不同，那是「同一解上的噪声」。

度量：逐种子算逐层净流入 `net_l = Σ_{h∈l} Δb_h`，然后看**三个种子之间的
逐层相关**。高相关 ⇒ 同一个解；低/负相关 ⇒ 不同的解。零 GPU。
"""
import os, sys, glob, argparse, numpy as np, torch, itertools
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.calib_scorer import CalibScorer                      # noqa: E402


def load(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    m = CalibScorer(sd.get("d_kv", 128), sd["L"], sd["H"],
                    n_slots=sd.get("slots", 8), d_m=sd.get("dim", 128),
                    mode="memoryless", arch=sd["arch"],
                    scale=sd.get("scale", "head"),
                    alpha_max=sd.get("alpha_max", 1.0))
    m.load_state_dict(sd["state"])
    return m.eval(), sd["L"], sd["H"]


@torch.no_grad()
def net_per_layer(m, L, H, traces, n_doc):
    net = np.zeros(L); neth = np.zeros(L*H)
    for f in sorted(glob.glob(os.path.join(ROOT, traces, "doc*.pt")))[:n_doc]:
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            g, t = float(ch["gsig"]), float(ch["thres"])
            S0, DS, RET, LH = [], [], [], []
            for l, pl in enumerate(ch["layers"]):
                s0 = pl["s0"].float(); mu, sh = pl["mu_h"].float(), pl["sig_h"].float()
                ds = m.delta(None, m.read((l,), None), s0,
                             margin=(s0 - t) / g, stats=(mu, sh, g))
                for h in range(s0.shape[0]):
                    S0.append(s0[h]); DS.append(ds[h]); RET.append(pl["ret"][h]); LH.append(l)
            S0 = torch.stack(S0); DS = torch.stack(DS); RET = torch.stack(RET)
            b0 = RET.sum(-1).numpy().astype(int); B = int(b0.sum())
            if B == 0:
                continue
            flat = (S0 + DS).reshape(-1)
            keep = torch.zeros_like(flat, dtype=torch.bool)
            keep[torch.topk(flat, B).indices] = True
            db = keep.reshape(S0.shape).sum(-1).numpy().astype(int) - b0
            for gi, l in enumerate(LH):
                net[l] += db[gi]; neth[gi] += db[gi]
    return net, neth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="scratch_ctrl_traces_v2_10")
    ap.add_argument("--n_doc", type=int, default=10)
    a = ap.parse_args()
    ARMS = [("chd10 有头嵌入", "varikv/chead10_s%d.pt/memoryless.pt", [24.40, 23.80, 24.40]),
            ("chd0  无头嵌入", "varikv/chead0_s%d.pt/memoryless.pt", [25.00, 21.40, 16.20])]
    for nm, pat, sc in ARMS:
        N = []; NH = []
        for s in (0, 1, 2):
            m, L, H = load(os.path.join(ROOT, pat % s))
            _n, _nh = net_per_layer(m, L, H, a.traces, a.n_doc)
            N.append(_n); NH.append(_nh)
        N = np.array(N); NH = np.array(NH)
        rs = [float(np.corrcoef(N[i], N[j])[0, 1]) for i, j in itertools.combinations(range(3), 2)]
        print(f"\n=== {nm}   分数 {sc}（跨度 {max(sc)-min(sc):.1f}）")
        print(f"  逐层净流入的**种子间相关**: " +
              " ".join(f"s{i}s{j}={r:+.3f}" for (i, j), r in
                       zip(itertools.combinations(range(3), 2), rs)) +
              f"   均值 **{np.mean(rs):+.3f}**")
        rh = [float(np.corrcoef(NH[i], NH[j])[0, 1]) for i, j in itertools.combinations(range(3), 2)]
        print(f"  **逐头**净流入的种子间相关: 均值 **{np.mean(rh):+.3f}**"
              f"（逐层是 {np.mean(rs):+.3f}）")
        for s in range(3):
            top = np.argsort(-N[s])[:3]; bot = np.argsort(N[s])[:3]
            print(f"  s{s}: 受主 top3 " + ",".join(f"L{l}({N[s][l]:+.0f})" for l in top) +
                  "   施主 top3 " + ",".join(f"L{l}({N[s][l]:+.0f})" for l in bot))
    print("\n判读：相关高 ⇒ 三个种子找到**同一个解**，分数差是别处来的；")
    print("      相关低 ⇒ **不同的解**，方差来自解的位置本身。")
