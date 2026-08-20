"""`c_h(x)` 到底有多「依赖上下文」？—— staticisation 的离线前置诊断。

论文卖点之一是「**状态条件**的跨头配额校准」。但如果 `c_h` 在所有 chunk 上几乎
不变，那学到的只是一张**与上下文无关的静态表**（「Qwen2.5 的 L1 该多给点」），
「状态条件」这个词就不能用。

本脚本只对 **chead 族**有效 —— 那里 `c_h` 就是逐 (chunk,层,头) 的一个标量，
可以直接做方差分解：

    Var_total = Var_between(头间，跨 chunk 取均值后) + E[Var_within(同一头跨 chunk)]

判据（先写下，不看结果）：
  * `within / total < 0.10`  ⇒ 基本是静态表，「状态条件」不成立，必须改写卖点；
  * `within / total > 0.40`  ⇒ 上下文依赖是主要成分，卖点成立；
  * 之间          ⇒ 两者都有，必须报比例、不能只说「动态」。

**⚠ 这只说明「c_h 变不变」，不说明「那点变化有没有用」。**
后者要靠真正的 staticisation 评测（把跨 chunk 均值冻成静态表重跑）来判。
"""
import os, sys, glob, argparse, numpy as np, torch
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.calib_scorer import CalibScorer                      # noqa: E402


def load(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    assert sd["arch"] == "chead", f"只对 chead 族有效，收到 arch={sd['arch']}"
    m = CalibScorer(sd.get("d_kv", 128), sd["L"], sd["H"],
                    n_slots=sd.get("slots", 8), d_m=sd.get("dim", 128),
                    mode="memoryless", arch=sd["arch"],
                    scale=sd.get("scale", "head"),
                    alpha_max=sd.get("alpha_max", 1.0))
    m.load_state_dict(sd["state"])
    return m.eval(), sd["L"], sd["H"], float(m.alpha)


@torch.no_grad()
def collect(m, L, H, traces, n_doc):
    """→ C[n_chunk, L*H]，单位 σ_g（决策相关的那个尺度）。"""
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, traces, "doc*.pt")))[:n_doc]:
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            g, t = float(ch["gsig"]), float(ch["thres"])
            r = []
            for l, pl in enumerate(ch["layers"]):
                s0 = pl["s0"].float()
                mu, sh = pl["mu_h"].float(), pl["sig_h"].float()
                mg = (s0 - t) / g
                ds = m.delta(None, m.read((l,), None), s0, margin=mg, stats=(mu, sh, g))
                # chead 头内恒定，取第 0 列即可；先自检确实恒定
                spread = (ds.max(-1).values - ds.min(-1).values).abs().max().item()
                assert spread < 1e-5, f"chead 头内不恒定！最大跨度 {spread:.2e}"
                r.append((ds[:, 0] / g).numpy())
            rows.append(np.concatenate(r))
    return np.array(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="scratch_ctrl_traces_v2_10")
    ap.add_argument("--n_doc", type=int, default=10)
    a = ap.parse_args()
    ARMS = [("chead  α=1.25", "varikv/chead_s%d.pt/memoryless.pt"),
            ("chd10  α=.999", "varikv/chead10_s%d.pt/memoryless.pt"),
            ("chead15 α=1.5", "varikv/chead15_s%d.pt/memoryless.pt")]
    print("`c_h` 的方差分解（单位 σ_g）。within = 同一头跨 chunk 的变动 = 上下文依赖\n")
    print(f"{'臂':<16}{'种子':>4}{'n_chunk':>8}{'sd_total':>10}{'sd_between':>11}"
          f"{'sd_within':>10}{'within/total':>13}")
    for nm, pat in ARMS:
        for s in (0, 1, 2):
            p = os.path.join(ROOT, pat % s)
            if not os.path.exists(p):
                continue
            m, L, H, al = load(p)
            C = collect(m, L, H, a.traces, a.n_doc)      # [n_chunk, G]
            mu_h = C.mean(0)                              # 每个头跨 chunk 的均值
            v_tot = C.var()                               # 全部 (chunk,头) 的总方差
            v_bet = mu_h.var()                            # 头间方差
            v_win = C.var(axis=0).mean()                  # 同头跨 chunk 方差的均值
            print(f"{nm:<16}{s:>4}{C.shape[0]:>8}{np.sqrt(v_tot):>10.4f}"
                  f"{np.sqrt(v_bet):>11.4f}{np.sqrt(v_win):>10.4f}"
                  f"{v_win/v_tot:>12.1%}")
