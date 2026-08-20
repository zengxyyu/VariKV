"""把 `chead` 学到的 `c_h` 冻成**静态逐头表**，产出一个 `bias`-架构 ckpt。

为什么用 `bias` 架构而不是新写一条通路：`bias` 的输出规范与 `chead` **逐字相同** ——
    chead: α·σ·tanh(mlp(feat_h))     （头内恒定，逐 chunk 变）
    bias : α·σ·tanh(ab[l,h,1])       （头内恒定，**逐 chunk 不变**）
所以只要令 `ab[l,h,1] = atanh( E_chunk[ tanh(mlp(feat_h)) ] )`，
静态臂在输出空间上就是动态臂的**逐 chunk 均值**，其余一切（界、尺度、α、
决策路径）逐位相同。**这是唯一变量为「是否随 chunk 变化」的对照。**

⚠ 取均值必须在 **tanh 之后**：`tanh(E[x]) ≠ E[tanh(x)]`，
在 `raw` 上取均值得到的不是输出均值（tanh 是凹/凸各半的非线性）。
"""
import os, sys, glob, argparse, numpy as np, torch
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.calib_scorer import CalibScorer                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="chead ckpt（memoryless.pt）")
    ap.add_argument("--out", required=True, help="输出目录，形如 varikv/stat_s0.pt")
    ap.add_argument("--traces", default="scratch_ctrl_traces_v2_10")
    ap.add_argument("--n_doc", type=int, default=10)
    a = ap.parse_args()

    sd = torch.load(a.src, map_location="cpu")
    assert sd["arch"] == "chead", f"源必须是 chead，收到 {sd['arch']}"
    # 只支持 global 尺度：chead 族全是 global，写一条未被覆盖的 head 分支
    # 只会增加出错面。要支持时再加，并同时加验收。
    assert sd.get("scale") == "global", f"只支持 scale=global，收到 {sd.get('scale')}"
    L, H = sd["L"], sd["H"]
    m = CalibScorer(sd.get("d_kv", 128), L, H, n_slots=sd.get("slots", 8),
                    d_m=sd.get("dim", 128), mode="memoryless", arch="chead",
                    scale=sd.get("scale", "head"),
                    alpha_max=sd["args"]["alpha_max"])
    m.load_state_dict(sd["state"]); m.eval()
    alpha = float(m.alpha)

    # 收集 tanh(raw)：从 delta 反解 —— delta = α·σ·tanh(raw) ⇒ tanh(raw) = delta/(α·σ)
    acc = np.zeros((L, H)); n = 0
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(ROOT, a.traces, "doc*.pt")))[:a.n_doc]:
            d = torch.load(f, map_location="cpu")
            for ch in d["chunks"]:
                g, t = float(ch["gsig"]), float(ch["thres"])
                for l, pl in enumerate(ch["layers"]):
                    s0 = pl["s0"].float()
                    mu, sh = pl["mu_h"].float(), pl["sig_h"].float()
                    mg = (s0 - t) / g
                    ds = m.delta(None, m.read((l,), None), s0,
                                 margin=mg, stats=(mu, sh, g))
                    acc[l] += (ds[:, 0] / (alpha * g)).numpy()
                n += 1
    th = acc / n                                   # E_chunk[tanh(raw)]，[L,H]
    assert np.abs(th).max() < 0.999, f"|E tanh| 逼近 1，atanh 会溢出：{np.abs(th).max()}"

    ab = torch.zeros(L, H, 2)
    ab[:, :, 1] = torch.from_numpy(np.arctanh(th)).float()   # bias 只用第 1 列
    out_sd = {"state": {"ab": ab, "alpha_on": sd["state"]["alpha_on"].clone()},
              "mode": "memoryless", "arch": "bias", "L": L, "H": H,
              "slots": sd.get("slots", 8), "dim": sd.get("dim", 128),
              "d_kv": sd.get("d_kv", 128), "scale": sd.get("scale", "head"),
              "args": dict(sd["args"], arch="bias",
                           _derived_from=os.path.relpath(a.src, ROOT))}
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    torch.save(out_sd, os.path.join(ROOT, a.out, "memoryless.pt"))

    # ---- 验收：静态臂的输出必须等于动态臂的逐 chunk 均值 ----
    m2 = CalibScorer(sd.get("d_kv", 128), L, H, n_slots=sd.get("slots", 8),
                     d_m=sd.get("dim", 128), mode="memoryless", arch="bias",
                     scale=sd.get("scale", "head"), alpha_max=sd["args"]["alpha_max"])
    m2.load_state_dict(out_sd["state"]); m2.eval()
    f0 = sorted(glob.glob(os.path.join(ROOT, a.traces, "doc*.pt")))[0]
    d0 = torch.load(f0, map_location="cpu"); ch0 = d0["chunks"][0]
    g0 = float(ch0["gsig"]); t0 = float(ch0["thres"]); pl0 = ch0["layers"][0]
    s0 = pl0["s0"].float(); mu, sh = pl0["mu_h"].float(), pl0["sig_h"].float()
    with torch.no_grad():
        ds2 = m2.delta(None, m2.read((0,), None), s0,
                       margin=(s0 - t0) / g0, stats=(mu, sh, g0))
    want = alpha * g0 * th[0]
    err = float(np.abs(ds2[:, 0].numpy() - want).max())
    spread = float((ds2.max(-1).values - ds2.min(-1).values).abs().max())
    print(f"写出 {a.out}/memoryless.pt  （bias 架构，225 参数，scale={out_sd['scale']}）")
    print(f"  α = {alpha:.4f}，静态 c_h 范围 [{(alpha*th).min():+.4f}, {(alpha*th).max():+.4f}] σ_g")
    print(f"  验收① 输出 == α·σ_g·E[tanh]：最大误差 {err:.2e}  {'OK' if err < 1e-5 else '**失败**'}")
    print(f"  验收② 头内恒定（保序）：最大跨度 {spread:.2e}  {'OK' if spread < 1e-5 else '**失败**'}")


if __name__ == "__main__":
    main()
