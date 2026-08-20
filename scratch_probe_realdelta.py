"""已训练各臂在真实 trace 上的**实际位移** |Δs|，以 σ_g 为单位。

`scratch_probe_ascale.py` 量的是**动作上界** α·σ；网络实际输出 tanh(raw) ∈ (−1,1)，
所以上界不等于实际位移。本脚本直接跑 `CalibScorer.delta`，量真值。
零 GPU。判据写成代码，判词由数字生成。
"""
import os, sys, glob, argparse, numpy as np, torch
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
    return m.eval(), sd["arch"], sd.get("scale", "head"), float(m.alpha)


@torch.no_grad()
def measure(m, traces, n_doc):
    """返回 (每头一个数的 |Δs|/σ_g 数组, 逐 token 的 |Δs|/σ_g 数组)。

    标量族头内逐 token 不同 ⇒ 两个口径都报：
      head 口径 = 每 (chunk,层,头) 的 mean|Δs|（与「动作半径」可比）
      tok  口径 = 全部 token 的 |Δs|
    """
    per_head, per_tok = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, traces, "doc*.pt")))[:n_doc]:
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            g, t = float(ch["gsig"]), float(ch["thres"])
            for l, pl in enumerate(ch["layers"]):
                s0 = pl["s0"].float()                       # [H, n]
                mu, sh = pl["mu_h"].float(), pl["sig_h"].float()
                mg = (s0 - t) / g
                r = m.read((l,), None)
                ds = m.delta(None, r, s0, margin=mg, stats=(mu, sh, g))
                a = (ds.abs() / g).numpy()                  # 以 σ_g 为单位
                per_head.append(a.mean(-1)); per_tok.append(a.ravel())
    return np.concatenate(per_head), np.concatenate(per_tok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="scratch_ctrl_traces_v2_10")
    ap.add_argument("--n_doc", type=int, default=10)
    a = ap.parse_args()

    ARMS = [("scalar σ_h α=.999", "varikv/d10_scalar_s%d.pt/memoryless.pt"),
            ("sgs    σ_g α=.999", "varikv/sg_scalar_s%d.pt/memoryless.pt"),
            ("sg125  σ_g α=1.25", "varikv/sg125_scalar_s%d.pt/memoryless.pt")]
    print(f"trace={a.traces}  n_doc={a.n_doc}   单位 = σ_g\n")
    print(f"{'臂':<20}{'α':>7}{'尺度':>8}"
          f"{'上界 α·E[σ]/σ_g':>18}{'实际 E|Δs|':>13}{'实际/上界':>11}{'p90|Δs|':>10}")
    rows = {}
    for name, pat in ARMS:
        hs, ts, al, sc, bnds = [], [], None, None, []
        for s in (0, 1, 2):
            p = os.path.join(ROOT, pat % s)
            if not os.path.exists(p):
                continue
            m, arch, sc, al = load(p)
            h, t = measure(m, a.traces, a.n_doc)
            hs.append(h.mean()); ts.append(t)
            # 上界：α·σ_h/σ_g（head 尺度）或 α（global 尺度）
            bnds.append(al)
        if not hs:
            print(f"{name:<20}  (缺 ckpt)"); continue
        tt = np.concatenate(ts)
        # 上界以 σ_g 为单位：global ⇒ α；head ⇒ α·E[σ_h]/σ_g
        if sc == "global":
            bnd = al
        else:
            A = []
            for f in sorted(glob.glob(os.path.join(ROOT, a.traces, "doc*.pt")))[:a.n_doc]:
                d = torch.load(f, map_location="cpu")
                for ch in d["chunks"]:
                    for pl in ch["layers"]:
                        A.append(pl["sig_h"].float().numpy() / float(ch["gsig"]))
            bnd = al * float(np.concatenate(A).mean())
        rows[name] = (np.mean(hs), bnd)
        print(f"{name:<20}{al:>7.3f}{sc:>8}{bnd:>18.4f}"
              f"{np.mean(hs):>13.4f}{np.mean(hs)/bnd:>11.3f}"
              f"{float(np.percentile(tt,90)):>10.4f}")

    if "scalar σ_h α=.999" in rows and "sgs    σ_g α=.999" in rows:
        a0, b0 = rows["scalar σ_h α=.999"]; a1, b1 = rows["sgs    σ_g α=.999"]
        print(f"\n=== 混淆的真实大小（σ_g 臂 vs σ_h 臂，同 α=0.999）")
        print(f"  上界比   {b1/b0:6.2f}×")
        print(f"  **实际位移比 {a1/a0:6.2f}×**")
        v = "实际位移几乎持平 ⇒ 混淆很小" if a1/a0 < 1.25 else (
            "实际位移也明显更大 ⇒ 混淆是真的，等幅度对照必须做")
        print(f"  ⇒ {v}")
