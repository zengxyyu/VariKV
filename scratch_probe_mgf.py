"""P0-D：高斯指数矩（MGF）可行性探针（NEXT_STEPS.md v5 §2.B.2 / §2.B.5）。

回答 v5 的 Q2 与 Q3：
  Q2  在真实 query 方向上，二阶高斯 MGF 近似准不准？
  Q3  它随上下文长度 N（固定 cluster 数 ⇒ 位置跨度 W 变大）是否系统性恶化？

诊断量必须在 **query 投影后的一维分布**上算，不是原始 key 峰度——softmax 只看到标量
`δ_i = aᵀ(k_i − μ_k)`，而 key 分布在 128 维里可以很不高斯、沿某个方向却近乎高斯。

一个化简让它非常便宜：
    δ_i = aᵀ(k_i − μ_k) = s_i − mean_cluster(s),   s_i = q·k_i/√d
所以不需要显式算 μ_k，直接把打分减去簇内均值即可。

    ε_MGF = log E_emp[e^δ] − ½·Var(δ) = κ₃/6 + κ₄/24 + …
    r_MGF = exp(ε_MGF)      r=1 精确；r=2 表示高斯把该簇分母低估 2 倍

同时报 δ 的偏度/峰度，看误差是否由高阶累积量解释。
cluster 用 **position-local**（v5 §2.B.3 的第一选择）：把被驱逐位置切成宽度 W 的连续块。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip                    # noqa: E402
from attention.kvcache import RetainCache               # noqa: E402
from data.load import load_dataset_all                  # noqa: E402
from data.wrapper import DataWrapper, get_query          # noqa: E402

_QCAP = {}
_orig_prepare = RetainCache.prepare


def _patched_prepare(self, query_states, key_states, value_states, layer_idx):
    _QCAP[layer_idx] = query_states.detach().clone()
    return _orig_prepare(self, query_states, key_states, value_states, layer_idx)


def get_valid(kv, layer_idx, S):
    try:
        v = kv._get_valid(layer_idx)
    except TypeError:
        v = kv._get_valid(layer_idx, S)
    if isinstance(v, (list, tuple)):
        v = torch.stack([x.bool() for x in v])
    v = v.bool()
    while v.dim() > 2:
        v = v.squeeze(0)
    if v.shape[-1] != S:
        pad = torch.ones(v.shape[0], S - v.shape[-1], dtype=torch.bool, device=v.device)
        v = torch.cat([v, pad], dim=-1)
    return v


@torch.no_grad()
def mgf_stats(kv, layer_idx, T, Ws, min_n=32, gq_max=1):
    """对最后一个 query token，逐 (kv_head, q_head, cluster) 算 ε_MGF / r_MGF / 偏度 / 峰度。

    返回 {W: dict(eps=[], r=[], skew=[], kurt=[], n=[])}
    """
    q = _QCAP.get(layer_idx)
    if q is None:
        return None
    k_all = kv.key_cache[layer_idx]
    S, H, d = k_all.shape[2], k_all.shape[1], k_all.shape[3]
    HQ = q.shape[1]; Gq = HQ // H
    dev = k_all.device
    valid = get_valid(kv, layer_idx, S).to(dev)
    scale = 1.0 / (d ** 0.5)
    res = {W: {"eps": [], "r": [], "skew": [], "kurt": [], "n": []} for W in Ws}

    for h in range(H):
        kh = k_all[0, h].float()                                 # [S,d]
        ev = (~valid[h]).nonzero(as_tuple=True)[0]               # 被驱逐位置
        if ev.numel() < min_n:
            continue
        for g in range(min(Gq, gq_max)):
            a = q[0].view(H, Gq, T, d)[h, g, -1].float()         # 末 token 的 query
            s_ev = (kh[ev] @ a) * scale                          # [n_E]  只算被驱逐的
            for W in Ws:
                blk = (ev // W)                                   # position-local 分簇
                for b in blk.unique():
                    sel = blk == b
                    n = int(sel.sum())
                    if n < min_n:
                        continue
                    x = s_ev[sel].double()
                    dl = x - x.mean()
                    var = dl.var(unbiased=False)
                    if var < 1e-12:
                        continue
                    eps = (torch.logsumexp(dl, 0) - np.log(n) - 0.5 * var).item()
                    z = dl / var.sqrt()
                    res[W]["eps"].append(eps)
                    res[W]["r"].append(float(np.exp(eps)))
                    res[W]["skew"].append(float((z ** 3).mean()))
                    res[W]["kurt"].append(float((z ** 4).mean() - 3.0))
                    res[W]["n"].append(n)
    return res


def rep(name, v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return f"  {name:<12} (空)"
    return (f"  {name:<12} n={v.size:<7d} median={np.median(v):9.4f}  "
            f"P10={np.quantile(v,.1):9.4f} P90={np.quantile(v,.9):9.4f}  "
            f"max={v.max():9.4f}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--gate", default="fastkvzip")
    ap.add_argument("--data", default="scbench_kv")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--chunk", type=int, default=16000)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--level", default="pair")
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--Ws", type=int, nargs="+", default=[512, 2048, 8192, 32768])
    ap.add_argument("--Ns", type=int, nargs="+", default=[16000, 32000, 64000, 128000])
    ap.add_argument("--layer_stride", type=int, default=4,
                    help="每隔几层取一层（控制 Python 层循环开销）")
    ap.add_argument("--gq_max", type=int, default=1,
                    help="每个 kv_head 取几个 query_head")
    args = ap.parse_args()

    RetainCache.prepare = _patched_prepare
    m = ModelKVzip(args.model, kv_type="retain", gate_path_or_name=args.gate)
    L = m.config.num_hidden_layers
    ds = load_dataset_all(args.data, m.tokenizer)
    dw = DataWrapper(args.data, ds, m)

    agg = {}      # (N, W) -> lists
    for si in range(args.n_samples):
        q0 = m.apply_template(get_query("qa", list(ds[si]["question"])[0]))
        full_ids = dw.prefill_context(si, do_score=False).ctx_ids.view(-1)
        torch.cuda.empty_cache()
        for N in args.Ns:
            if full_ids.numel() < N:
                continue
            ids = full_ids[-N:].view(1, -1).to(m.device)
            _QCAP.clear()
            kv = m.prefill(ids, prefill_chunk_size=args.chunk, do_score=True,
                           chunk_ratio=args.ratio, window_size=args.window,
                           level=args.level)
            m.model(q0.to(m.device), past_key_values=kv)          # 触发 prepare，拿 query
            T = q0.shape[-1]
            for l in range(0, L, args.layer_stride):
                r = mgf_stats(kv, l, T, args.Ws, gq_max=args.gq_max)
                if r is None:
                    continue
                for W, dd in r.items():
                    a = agg.setdefault((N, W), {k: [] for k in dd})
                    for k, v in dd.items():
                        a[k].extend(v)
            print(f"样本{si} N={N}: 已累计 "
                  + " ".join(f"W{W}:{len(agg.get((N,W),{'r':[]})['r'])}"
                             for W in args.Ws), flush=True)
            del kv; torch.cuda.empty_cache()

    print("\n" + "=" * 92)
    print("r_MGF = E_emp[e^δ] / e^{Var(δ)/2}   （1 = 二阶高斯 MGF 精确；>1 = 低估分母）")
    for N in args.Ns:
        if not any((N, W) in agg for W in args.Ws):
            continue
        print(f"\n上下文 N = {N}")
        for W in args.Ws:
            a = agg.get((N, W))
            if not a or not a["r"]:
                continue
            r = np.array(a["r"]); e = np.array(a["eps"])
            sk = np.array(a["skew"]); ku = np.array(a["kurt"])
            nn = np.array(a["n"])
            print(f"  W={W:<6} 簇数={r.size:<6d} n/簇中位={np.median(nn):7.0f}  "
                  f"r 中位={np.median(r):7.3f}  r P90={np.quantile(r,.9):8.3f}  "
                  f"|ε|中位={np.median(np.abs(e)):7.4f}  "
                  f"偏度中位={np.median(sk):+6.2f}  超峰度中位={np.median(ku):+7.2f}")
    print("=" * 92)
    print("判读（v5 Q2/Q3）：")
    print("  r 中位 ≈ 1 且随 N 稳定 ⇒ 二阶分布式摘要成立，可以做 E3/E4。")
    print("  r 明显偏离 1，但偏度/超峰度大 ⇒ 需要三阶/四阶累积量（研究问题升级）。")
    print("  r 随 N 或 W 系统性恶化 ⇒ 固定 K 的高斯 cluster 无长度可扩展性 ⇒ 走 §2.B.5 路线 D。")


if __name__ == "__main__":
    main()
