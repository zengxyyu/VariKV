"""§15 判据探针：**层级净量**比**逐头之和**好压缩吗？（免训练，不跑下游）

起因（P0，2026-08-11）：`‖Σ_h W_O Δo_h‖ / Σ_h ‖W_O Δo_h‖ = 0.253` ——
75% 的逐头驱逐损伤在层内自我抵消，而 ResKV / IndexMem / Tensor Cache / 我们
全都是**逐头独立**修正的。实测后果：部分恢复（top-80 of 784）把某些样本的 `B`
从 3.32 推到 6.44（比不修更糟），`all` 才始终安全。

本探针只回答**前提问题**，不预设参数化：

    在残差流里，层级净量  Y(q) = Σ_h W_O^{(h)} Δo_h(q)  ∈ R^{d_model}
    是否比"各头分量的并集" {P_h(q) = W_O^{(h)} Δo_h(q)} 显著更低维？

三个量，一个对照：

  cancel     = ‖Y‖_F / Σ_h‖P_h‖_F                     复现 0.253（逐层）
  rank90(Y)  vs  rank90([P_1|…|P_H])                  净量 vs 各头并集的维度
  **shuffle 对照**：把每个头的行（query）各自独立乱序后再求和。
             若 rank90(Y_shuf) ≈ rank90(Y)，那么 Y 的低维只是"求和总是更简单"，
             **不是结构性相消** ⇒ §15 没有额外可利用的东西。这是本脚本最重要的一栏。

外加一个等字节重建对比（次要，偏差已知）：

  err_joint(r) = ‖Y − trunc_r(Y)‖/‖Y‖              存 r 个 R^{d_model} 基 = r·d_model 标量
  err_head(ρ)  = ‖Y − Σ_h trunc_ρ(P_h)‖/‖Y‖        每头存 ρ 个 R^{d_head} 基 = ρ·d_model 标量
  ⇒ **等字节 ⟺ r = ρ**，但逐头方案此时拿到 H_q·ρ 个方向、联合方案只有 r 个。
  这个记账**对逐头有利**（方向数多 28 倍），所以它不是判据，只是参考；
  判据看 shuffle 对照与 rank90 之比。

用法：
    .venv/bin/python scratch_probe_layerjoint.py --n_samples 3 --n_queries 4
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip                     # noqa: E402
from attention.kvcache import RetainCache                # noqa: E402
from data.load import load_dataset_all                   # noqa: E402
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
def delta_o_all(kv, layer_idx, T):
    """Δo_h = o_all − o_R，同一 q/K/V 内的精确反事实。返回 [HQ, T, d]。"""
    q = _QCAP[layer_idx]
    k_all, v_all = kv.key_cache[layer_idx], kv.value_cache[layer_idx]
    S, H, d = k_all.shape[2], k_all.shape[1], k_all.shape[3]
    HQ = q.shape[1]
    Gq = HQ // H
    dev = k_all.device
    valid = get_valid(kv, layer_idx, S).to(dev)
    idx_k = torch.arange(S, device=dev).view(1, S)
    idx_q = (S - T) + torch.arange(T, device=dev).view(T, 1)
    causal = idx_k <= idx_q
    neg = torch.finfo(torch.float32).min
    scale = 1.0 / (d ** 0.5)
    out = torch.empty(HQ, T, d, device=dev, dtype=torch.float32)
    for hq in range(HQ):
        h, g = hq // Gq, hq % Gq
        qh = q[0].view(H, Gq, T, d)[h, g].float()
        kh = k_all[0, h].float()
        vh = v_all[0, h].float()
        s = (qh @ kh.T) * scale
        s = s.masked_fill(~causal, neg)
        vm = valid[h].view(1, S)
        o_R = torch.softmax(s.masked_fill(~vm, neg), -1) @ vh
        o_A = torch.softmax(s, -1) @ vh
        out[hq] = o_A - o_R
        del s, o_R, o_A
    return out


def spec(M):
    """返回奇异值（降序），走 Gram 矩阵（行数 ≪ 列数）。"""
    G = (M @ M.T).double()
    ev = torch.linalg.eigvalsh(G).flip(0).clamp_min(0)
    return ev.sqrt()


def rank_at(sv, frac):
    e = (sv ** 2).cumsum(0)
    if float(e[-1]) <= 0:
        return 0
    return int((e / e[-1] < frac).sum()) + 1


def part_ratio(sv):
    """participation ratio：(Σσ²)² / Σσ⁴，连续版有效秩。"""
    s2 = (sv ** 2)
    return float(s2.sum() ** 2 / (s2 ** 2).sum().clamp_min(1e-30))


def trunc_err(M, r):
    sv = spec(M)
    tot = float((sv ** 2).sum())
    if tot <= 0:
        return 0.0
    kept = float((sv[:r] ** 2).sum())
    return float(np.sqrt(max(tot - kept, 0.0) / tot))


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
    ap.add_argument("--n_queries", type=int, default=4)
    ap.add_argument("--layers", type=int, nargs="*",
                    default=[0, 6, 13, 20, 24, 26, 27])
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 2, 4, 8])
    args = ap.parse_args()

    RetainCache.prepare = _patched_prepare
    m = ModelKVzip(args.model, kv_type="retain", gate_path_or_name=args.gate)
    HQ = m.config.num_attention_heads
    d = getattr(m.config, "head_dim", m.config.hidden_size // HQ)
    dmodel = m.config.hidden_size
    print(f"[cfg] d_model={dmodel} HQ={HQ} d_head={d} ratio={args.ratio}", flush=True)

    WO = {}
    for l in args.layers:
        W = m.model.model.layers[l].self_attn.o_proj.weight.detach().float()
        WO[l] = W                                        # [d_model, HQ*d]

    ds = load_dataset_all(args.data, m.tokenizer)
    dw = DataWrapper(args.data, ds, m)

    rows = defaultdict(list)                             # (si, l) -> [HQ, Q, d]
    for si in range(args.n_samples):
        qs = list(ds[si]["question"])[: args.n_queries]
        qid = [m.apply_template(get_query("qa", q)) for q in qs]
        kv = dw.prefill_context(si, prefill_chunk=args.chunk,
                                window_size=args.window,
                                chunk_ratio=args.ratio, level=args.level)
        S = kv.key_cache[0].shape[2]
        for ids in qid:
            _QCAP.clear()
            m.model(ids.to(m.device), past_key_values=kv)
            T = ids.shape[-1]
            for l in args.layers:
                rows[(si, l)].append(delta_o_all(kv, l, T).cpu())
            kv.slice(S)
        del kv
        torch.cuda.empty_cache()
        print(f"[样本{si}] Q={sum(x.shape[1] for x in rows[(si, args.layers[0])])}",
              flush=True)

    g = torch.Generator().manual_seed(0)
    per_layer = defaultdict(list)
    for si in range(args.n_samples):
        for l in args.layers:
            D = torch.cat(rows[(si, l)], dim=1)          # [HQ, Q, d]
            Q = D.shape[1]
            W = WO[l].cpu()
            P = [D[h] @ W[:, h * d:(h + 1) * d].T for h in range(HQ)]   # [Q,d_model]
            Y = torch.stack(P).sum(0)
            sum_norm = float(sum(float(p.norm()) for p in P))
            cancel = float(Y.norm()) / max(sum_norm, 1e-30)

            Ys = torch.zeros_like(Y)
            for p in P:
                Ys += p[torch.randperm(Q, generator=g)]

            sv_Y, sv_S = spec(Y), spec(Ys)
            svc = torch.linalg.eigvalsh(
                sum((p @ p.T).double() for p in P)).flip(0).clamp_min(0).sqrt()

            eh, ej = {}, {}
            for r in args.ranks:
                ej[r] = trunc_err(Y, r)
                acc = torch.zeros_like(Y)
                for p in P:
                    U, S_, Vt = torch.linalg.svd(p, full_matrices=False)
                    acc += (U[:, :r] * S_[:r]) @ Vt[:r]
                eh[r] = float((Y - acc).norm() / Y.norm().clamp_min(1e-30))

            per_layer[l].append(dict(
                Q=Q, cancel=cancel,
                r90Y=rank_at(sv_Y, .90), r90S=rank_at(sv_S, .90),
                r90C=rank_at(svc, .90),
                prY=part_ratio(sv_Y), prS=part_ratio(sv_S), prC=part_ratio(svc),
                ej=ej, eh=eh))

    print("\n" + "=" * 116)
    print("逐层（多样本中位数）。r90 = 占 90% 能量所需秩；PR = participation ratio")
    print(f"{'层':>4}{'Q':>5}{'cancel':>8}{'r90(Y)':>8}{'r90(shuf)':>10}"
          f"{'r90(并集)':>10}{'PR(Y)':>8}{'PR(shuf)':>9}{'PR(并集)':>9}"
          f"{'  |  等字节 err_joint / err_head':>34}")
    agg = defaultdict(list)
    for l in args.layers:
        R = per_layer[l]
        md = lambda k: float(np.median([x[k] for x in R]))                # noqa: E731
        eb = "  ".join(
            f"r={r}:{np.median([x['ej'][r] for x in R]):.3f}/"
            f"{np.median([x['eh'][r] for x in R]):.3f}" for r in args.ranks)
        print(f"{l:>4}{R[0]['Q']:>5}{md('cancel'):>8.3f}{md('r90Y'):>8.0f}"
              f"{md('r90S'):>10.0f}{md('r90C'):>10.0f}{md('prY'):>8.1f}"
              f"{md('prS'):>9.1f}{md('prC'):>9.1f}   {eb}")
        for k in ("cancel", "r90Y", "r90S", "r90C", "prY", "prS", "prC"):
            agg[k].append(md(k))
        for r in args.ranks:
            agg[f"ej{r}"].append(float(np.median([x["ej"][r] for x in R])))
            agg[f"eh{r}"].append(float(np.median([x["eh"][r] for x in R])))
    print("-" * 116)
    print(f"{'均值':>4}{'':>5}{np.mean(agg['cancel']):>8.3f}"
          f"{np.mean(agg['r90Y']):>8.1f}{np.mean(agg['r90S']):>10.1f}"
          f"{np.mean(agg['r90C']):>10.1f}{np.mean(agg['prY']):>8.1f}"
          f"{np.mean(agg['prS']):>9.1f}{np.mean(agg['prC']):>9.1f}   "
          + "  ".join(f"r={r}:{np.mean(agg[f'ej{r}']):.3f}/"
                      f"{np.mean(agg[f'eh{r}']):.3f}" for r in args.ranks))
    print("=" * 116)
    rY, rS = np.mean(agg["r90Y"]), np.mean(agg["r90S"])
    rC = np.mean(agg["r90C"])
    print(f"判读① 结构性相消：r90(Y)/r90(shuf) = {rY / max(rS, 1e-9):.3f}  "
          "（≪1 ⇒ 低维来自 query 对齐的相消，是真结构；≈1 ⇒ 只是'求和更简单'，§15 停）")
    print(f"判读② 净量 vs 并集： r90(Y)/r90(并集) = {rY / max(rC, 1e-9):.3f}  "
          "（≪1 ⇒ 建模净量本质更省）")
    print(f"判读③ 等字节：err_joint ≲ ½·err_head 才算方向成立（此记账偏向逐头 {HQ}×）")


if __name__ == "__main__":
    main()
