"""ε_MGF 对 query 方向的敏感性（P0_FINDINGS §3 提出的下一个待测量）。

动机：P0-D 测出高斯二阶 MGF 的中位误差是 4–14%、最差十分位 70–170%，而误差由簇内投影打分的
**离散度 σ** 驱动（超峰度≈0，不是重尾）。如果 `ε_MGF` 对 query 方向不敏感，那么**每簇存一个
log 校正标量**（1 个 scalar/簇，几乎免费）就能把误差压掉——这会让高斯路线便宜很多。

做法：对同一个 cluster，用**多个真实 query 方向**（不同问题的末 token × 不同 query head）
各算一次 `ε_MGF`，比较
    簇内跨 query 的标准差   std_within
    跨簇的标准差           std_across
若 `std_within ≪ std_across` ⇒ ε 主要是簇的属性 ⇒ 存一个标量可行。
若两者相当 ⇒ ε 是 (簇, query) 的联合属性 ⇒ 存标量无用，必须在读出时算。
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
def eps_matrix(kv, layer_idx, W, dirs, min_n=64, max_clusters=64):
    """返回 [n_cluster, n_dir] 的 ε_MGF 矩阵。dirs: list of (h, a) 其中 a 是 [d] 的 query 方向。"""
    k_all = kv.key_cache[layer_idx]
    S, H, d = k_all.shape[2], k_all.shape[1], k_all.shape[3]
    valid = get_valid(kv, layer_idx, S).to(k_all.device)
    scale = 1.0 / (d ** 0.5)
    per_h = {}
    for h, a in dirs:
        per_h.setdefault(h, []).append(a)
    rows = []
    for h, alist in per_h.items():
        kh = k_all[0, h].float()
        ev = (~valid[h]).nonzero(as_tuple=True)[0]
        if ev.numel() < min_n:
            continue
        blk = ev // W
        ub = blk.unique()[:max_clusters]
        A = torch.stack(alist).float()                      # [n_dir, d]
        s_all = (kh[ev] @ A.T) * scale                      # [n_E, n_dir]
        for b in ub:
            sel = blk == b
            if int(sel.sum()) < min_n:
                continue
            x = s_all[sel].double()                         # [n, n_dir]
            dl = x - x.mean(0, keepdim=True)
            var = dl.var(0, unbiased=False)                 # [n_dir]
            if (var < 1e-12).any():
                continue
            eps = (torch.logsumexp(dl, 0) - np.log(int(sel.sum())) - 0.5 * var)
            rows.append(eps.cpu().numpy())
    return np.array(rows) if rows else None


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
    ap.add_argument("--N", type=int, default=64000)
    ap.add_argument("--Ws", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--n_samples", type=int, default=2)
    ap.add_argument("--n_queries", type=int, default=4, help="用几个不同问题的末 token 作方向")
    ap.add_argument("--gq", type=int, default=2, help="每 kv_head 取几个 query_head 作方向")
    ap.add_argument("--layer_stride", type=int, default=6)
    args = ap.parse_args()

    RetainCache.prepare = _patched_prepare
    m = ModelKVzip(args.model, kv_type="retain", gate_path_or_name=args.gate)
    L = m.config.num_hidden_layers
    H = m.config.num_key_value_heads
    Gq = m.config.num_attention_heads // H
    ds = load_dataset_all(args.data, m.tokenizer)
    dw = DataWrapper(args.data, ds, m)

    agg = {W: {"within": [], "across": [], "mean": []} for W in args.Ws}

    for si in range(args.n_samples):
        qs = [m.apply_template(get_query("qa", q))
              for q in list(ds[si]["question"])[: args.n_queries]]
        full_ids = dw.prefill_context(si, do_score=False).ctx_ids.view(-1)
        torch.cuda.empty_cache()
        ids = full_ids[-args.N:].view(1, -1).to(m.device)
        kv = m.prefill(ids, prefill_chunk_size=args.chunk, do_score=True,
                       chunk_ratio=args.ratio, window_size=args.window,
                       level=args.level)
        # 逐问题跑一次前向，收集各层的末 token query 作为方向
        dirs_by_layer = {l: [] for l in range(0, L, args.layer_stride)}
        S0 = kv.key_cache[0].shape[2]
        for qi in qs:
            _QCAP.clear()
            m.model(qi.to(m.device), past_key_values=kv)
            for l in dirs_by_layer:
                q = _QCAP.get(l)
                if q is None:
                    continue
                T = q.shape[2]
                for h in range(H):
                    for g in range(min(Gq, args.gq)):
                        dirs_by_layer[l].append((h, q[0].view(H, Gq, T, -1)[h, g, -1]))
            kv.slice(S0)
        for l, dirs in dirs_by_layer.items():
            if not dirs:
                continue
            for W in args.Ws:
                E = eps_matrix(kv, l, W, dirs)
                if E is None or E.shape[1] < 2:
                    continue
                # 每个 h 的方向数 = n_queries * gq；同 h 内跨方向 = 簇内跨 query 变异
                agg[W]["within"].append(E.std(axis=1))
                agg[W]["across"].append(np.full(E.shape[0], E.mean(axis=1).std()))
                agg[W]["mean"].append(E.mean(axis=1))
        print(f"样本{si}: " + " ".join(
            f"W{W}:{sum(len(x) for x in agg[W]['within'])}簇" for W in args.Ws), flush=True)
        del kv; torch.cuda.empty_cache()

    print("\n" + "=" * 88)
    print("ε_MGF 的方差分解：簇内跨 query  vs  跨簇")
    for W in args.Ws:
        a = agg[W]
        if not a["within"]:
            continue
        wi = np.concatenate(a["within"]); ac = np.concatenate(a["across"])
        mu = np.concatenate(a["mean"])
        print(f"  W={W:<6} 簇数={wi.size:<6d}  ε均值中位={np.median(mu):8.4f}  "
              f"簇内 std 中位={np.median(wi):8.4f}  跨簇 std={np.median(ac):8.4f}  "
              f"比值 within/across={np.median(wi)/max(np.median(ac),1e-12):6.2f}")
    print("=" * 88)
    print("判读：比值 ≪1 ⇒ ε 主要是簇属性 ⇒ **每簇存一个 log 校正标量可行**（几乎免费）。")
    print("      比值 ≳1 ⇒ ε 是 (簇, query) 联合属性 ⇒ 存标量无用，必须读出时现算。")


if __name__ == "__main__":
    main()
