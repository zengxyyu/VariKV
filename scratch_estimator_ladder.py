"""P1：免训练的统计估计器阶梯 E0→E5（NEXT_STEPS.md v5 §5、P0_FINDINGS §2.1）。

问题（H1）：在**相同比特预算**下，把预算花在更多点原型上更好，还是更少但带协方差的分布式摘要？

做法：复用 P0-B' 的 intervention 机制——在 `o_proj` 的 forward pre-hook 里注入
`Δ̂o = ô_all − o_R`，其中

    ô_all = (N_R + N̂_E) / (D_R + D̂_E)        ← v5 §1.1 的精确代数

`N_R, D_R` 从保留缓存精确算，`N̂_E, D̂_E` 由各估计器从**有界摘要**给出。
判据 = 恢复了多少 `B = KL(p_full‖p_pruned)`，与 oracle（E0，实测 −55.7%）直接可比。

**两条 P0 得到的硬约束，已写进实现：**
1. **层内全修或不修**：所有 head 一起注入。部分恢复会打破 75% 的跨头相消，
   实测能把 B 从 3.32 推到 6.44。
2. **equal-bytes 而非 same-K**：高斯簇约 5.5× 点簇的存储，same-K 比较不成立。

数值：一切都相对 `e^{M0}`（M0 = 该 (层,头,query) 上所有 score 的最大值）归一化，
所以 `ô_all` 与 M0 无关，169k 上不会溢出。

阶梯：
  E0    精确 (N_E, D_E)，用全量被驱逐集 —— oracle 天花板
  E1    点质心 (n, μ_k, μ_v)，same-K
  E1b   点质心，**equal-bytes**（更多簇）
  E2    MomentKV 式一阶：分母同 E1，分子加 Σ_vk·a（低秩）
  E3    高斯，仅 diag(Σ_kk)（只改分母）
  E4    高斯，分子+分母（Σ_kk 与低秩 Σ_vk）
  E4c   E4 + **oracle 的逐簇 log 校正 ε_c**（上界：一个存储标量最多能买到多少）
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

from model.wrapper import ModelKVzip                    # noqa: E402
from attention.kvcache import RetainCache               # noqa: E402
from data.load import load_dataset_all                  # noqa: E402
from data.wrapper import DataWrapper, get_query          # noqa: E402

_QCAP = {}
_orig_prepare = RetainCache.prepare
_ST = {"kv": None, "arm": None, "sum": None, "d": None, "on": False}
_CLIP = [0, 0]        # [被钳位的项数, 总项数]


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


# ---------------------------------------------------------------- 写入：建摘要
@torch.no_grad()
def build_summaries(kv, L, H, Ws, r_c, r_k):
    """逐 (layer, kv_head) 建各估计器的摘要。摘要与 query 无关，只建一次。

    Ws: {"gauss": W_g, "point": W_p}  ——  W_p 更小以匹配 equal-bytes
    返回 {(l,h): {"ev": idx, "g": {...}, "p": {...}, "pk": {...}}}
    """
    out = {}
    for l in range(L):
        k_all, v_all = kv.key_cache[l], kv.value_cache[l]
        S = k_all.shape[2]
        valid = get_valid(kv, l, S).to(k_all.device)
        for h in range(H):
            ev = (~valid[h]).nonzero(as_tuple=True)[0]
            if ev.numel() < 8:
                out[(l, h)] = None
                continue
            kh = k_all[0, h, ev].float()               # [n_E, d]
            vh = v_all[0, h, ev].float()
            rec = {"ev": ev}
            for tag, W in (("g", Ws["gauss"]), ("p", Ws["point"])):
                blk = ev // W
                ub = blk.unique()
                n_list, mk, mv, vk, cv, uk, dres = [], [], [], [], [], [], []
                for b in ub:
                    sel = blk == b
                    n = int(sel.sum())
                    K_, V_ = kh[sel], vh[sel]
                    mu_k = K_.mean(0); mu_v = V_.mean(0)
                    n_list.append(n); mk.append(mu_k); mv.append(mu_v)
                    dk = K_ - mu_k
                    vk.append((dk * dk).mean(0))                       # diag Σ_kk
                    if tag == "g" and r_k > 0 and n > 1:
                        # **对角 Σ_kk 不可用**：它把 Σ_j a_j²σ_j² 当作投影方差，忽略 key
                        # 各维相关，交叉项本该抵消 ⇒ 严重高估 ⇒ exp 溢出（实测 E3/E4 出 NaN）。
                        # 存 diag + rank-r_k：Var(aᵀδ) ≈ Σ_j a_j² d_j + ‖U_kᵀa‖²
                        C = (dk.T @ dk) / n
                        evals, evecs = torch.linalg.eigh(C.double())
                        r = min(r_k, evals.numel())
                        Uk = (evecs[:, -r:] * evals[-r:].clamp_min(0).sqrt()).float()
                        uk.append(Uk)
                        dres.append((C.diag().float() - (Uk * Uk).sum(-1)).clamp_min(0))
                    if tag == "g":
                        dv = V_ - mu_v
                        Svk = (dv.T @ dk) / max(n, 1)                  # [d,d]
                        if r_c > 0 and n > 1:
                            U, Sg, Vt = torch.linalg.svd(Svk.double(), full_matrices=False)
                            r = min(r_c, Sg.numel())
                            cv.append(((U[:, :r] * Sg[:r]).float(), Vt[:r].float()))
                        else:
                            cv.append((torch.zeros(V_.shape[1], 1, device=K_.device),
                                       torch.zeros(1, K_.shape[1], device=K_.device)))
                rec[tag] = {
                    "blk": blk, "ub": ub,
                    "n": torch.tensor(n_list, dtype=torch.float32, device=kh.device),
                    "mu_k": torch.stack(mk), "mu_v": torch.stack(mv),
                    "var_k": torch.stack(vk),
                    "cv": cv if tag == "g" else None,
                    "Uk": uk if (tag == "g" and uk) else None,
                    "dres": torch.stack(dres) if (tag == "g" and dres) else None,
                }
            out[(l, h)] = rec
    return out


# ---------------------------------------------------------------- 读出：各估计器
@torch.no_grad()
def estimate(arm, rec, a, kh_ev, vh_ev, M0):
    """返回 (N_scaled [T,d], D_scaled [T], LM [T])，真值 = 该量 × e^{M0+LM}。

    **为什么要再引入 LM**：被驱逐簇的 logD 可以很大（实测 M 的 P99 = 0.965，
    存在驱逐集完全主导的 (层,头,token) 点），float32 下 exp 溢出成 inf，
    随后 inf/inf → nan（2026-08-12 实测 E3/E4 全 NaN 就是这个原因，
    **不是**对角协方差高估——对角实测是真值的 0.90 倍，够用）。
    """
    T = a.shape[0]
    if arm == "E0":                                     # 精确
        s = a @ kh_ev.T                                 # [T,n_E]
        lg = s - M0[:, None]
        LM = lg.max(-1).values.clamp_min(0.0)
        w = torch.exp(lg - LM[:, None])
        return w @ vh_ev, w.sum(-1), LM

    tag = "p" if arm in ("E1b",) else "g"
    if arm == "E1":
        tag = "g"                                       # same-K：与高斯同簇宽
    S_ = rec[tag]
    lm = a @ S_["mu_k"].T                               # [T,K] = a·μ_k
    if arm in ("E3", "E4", "E4c"):
        if S_.get("Uk") is not None:                    # diag(残差) + 低秩
            base = (a * a) @ S_["dres"].T               # [T,K]
            lr = torch.stack([((a @ U) ** 2).sum(-1) for U in S_["Uk"]], -1)
            quad = 0.5 * (base + lr)
        else:                                           # 纯对角（已知会高估，仅作对照）
            quad = 0.5 * (a * a) @ S_["var_k"].T
        lm = lm + quad
    if arm == "E4c":
        lm = lm + S_.get("eps", 0.0)                    # oracle 逐簇 log 校正
    logD = lm + torch.log(S_["n"])[None, :] - M0[:, None]
    # 钳位：保守起见挡住溢出（若近似把方差高估，exp 会炸）。统计钳位比例。
    LM = logD.max(-1).values.clamp_min(0.0)             # 逐 token 的公共尺度
    Dc = torch.exp(logD - LM[:, None])                  # [T,K]，最大项 ≤ 1
    mu_v = S_["mu_v"]                                   # [K,d]
    if arm in ("E1", "E1b", "E3"):
        Nc = Dc @ mu_v
        return Nc, Dc.sum(-1), LM
    else:                                               # E2/E4/E4c：分子加 Σ_vk·a
        corr = torch.zeros(T, mu_v.shape[0], mu_v.shape[1], device=a.device)
        for j, (U, Vt) in enumerate(S_["cv"]):
            corr[:, j, :] = (a @ Vt.T) @ U.T            # Σ_vk a = U(Vᵀa)
        Nc = torch.einsum("tk,tkd->td", Dc, mu_v[None].expand(T, -1, -1) + corr)
    return Nc, Dc.sum(-1), LM


@torch.no_grad()
def delta_o_est(kv, layer_idx, T, arm, summaries):
    """返回 {hq: Δ̂o [T,d]}，对**全部** head（层内全修或不修）。"""
    q = _QCAP.get(layer_idx)
    if q is None:
        return {}
    k_all, v_all = kv.key_cache[layer_idx], kv.value_cache[layer_idx]
    S, H, d = k_all.shape[2], k_all.shape[1], k_all.shape[3]
    HQ = q.shape[1]; Gq = HQ // H
    dev = k_all.device
    valid = get_valid(kv, layer_idx, S).to(dev)
    idx_k = torch.arange(S, device=dev).view(1, S)
    idx_q = (S - T) + torch.arange(T, device=dev).view(T, 1)
    causal = idx_k <= idx_q
    neg = torch.finfo(torch.float32).min
    scale = 1.0 / (d ** 0.5)
    out = {}
    for h in range(H):
        rec = summaries.get((layer_idx, h))
        if rec is None:
            continue
        kh = k_all[0, h].float(); vh = v_all[0, h].float()
        ev = rec["ev"]
        kh_ev, vh_ev = kh[ev], vh[ev]
        for g in range(Gq):
            hq = h * Gq + g
            a = q[0].view(H, Gq, T, d)[h, g].float() * scale            # [T,d]
            s = a @ kh.T                                                # [T,S]
            s = s.masked_fill(~causal, neg)
            sR = s.masked_fill(~valid[h].view(1, S), neg)
            M0 = sR.max(-1).values                                      # 归一化基准
            wR = torch.exp(sR - M0[:, None])
            DR = wR.sum(-1); NR = wR @ vh                               # 相对 e^{M0}
            o_R = NR / DR.clamp_min(1e-30)[:, None]
            if arm == "E4c":                                            # oracle 校正
                for tag in ("g",):
                    S_ = rec[tag]
                    eps = []
                    for b in S_["ub"]:
                        sel = S_["blk"] == b
                        x = (a @ kh_ev[sel].T).double()                 # [T,n]
                        dl = x - x.mean(-1, keepdim=True)
                        var = dl.var(-1, unbiased=False)
                        eps.append((torch.logsumexp(dl, -1)
                                    - np.log(int(sel.sum())) - 0.5 * var).float())
                    S_["eps"] = torch.stack(eps, -1)                    # [T,K]
            NE, DE, LM = estimate(arm, rec, a, kh_ev, vh_ev, M0)
            sc = torch.exp(-LM)                                          # 把保留侧也缩到同尺度
            o_hat = ((NR * sc[:, None] + NE)
                     / (DR * sc + DE).clamp_min(1e-30)[:, None])
            out[hq] = o_hat - o_R
            del s, sR, wR
    torch.cuda.empty_cache()
    return out


def make_hook(layer_idx):
    def pre_hook(module, args):
        if not _ST["on"]:
            return None
        x = args[0]
        T = x.shape[1]; d = _ST["d"]
        deltas = delta_o_est(_ST["kv"], layer_idx, T, _ST["arm"], _ST["sum"])
        if not deltas:
            return None
        x = x.clone()
        for hq, dl in deltas.items():
            x[0, :, hq * d:(hq + 1) * d] += dl.to(x.dtype)
        return (x,) + tuple(args[1:])
    return pre_hook


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
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--n_queries", type=int, default=3)
    ap.add_argument("--W_gauss", type=int, default=8192)
    ap.add_argument("--r_c", type=int, default=4, help="Σ_vk 的低秩秩数")
    ap.add_argument("--r_k", type=int, default=0,
                    help="Σ_kk 的低秩秩数。**实测 0（纯对角）就够**：对角给出真实投影方差的 "
                         "0.90 倍（r8 → 0.97），见 scratch_probe_cov_rank.py")
    ap.add_argument("--arms", nargs="+",
                    default=["none", "E0", "E1", "E1b", "E2", "E3", "E4", "E4c"])
    args = ap.parse_args()

    # equal-bytes：高斯簇 = 1 + 2d + d + 2·d·r_c + 1；点簇 = 1 + 2d
    d0 = 128
    # n(1) + μ_k,μ_v(2d) + diag 残差(d) + Σ_kk 低秩(d·r_k) + Σ_vk 低秩(2d·r_c) + ε(1)
    # 实测（scratch_probe_cov_rank.py）：对角 Σ_kk 已给出真实投影方差的 0.90 倍，
    # 加秩只微调（r8→0.97, r16→0.99），所以**不需要低秩 Σ_kk**，预算里只算对角。
    b_g = 1 + 2 * d0 + d0 + d0 * args.r_k + 2 * d0 * args.r_c + 1
    b_p = 1 + 2 * d0
    W_point = max(256, int(args.W_gauss * b_p / b_g))
    print(f"[预算] 高斯簇 {b_g} scalars/簇（r_k={args.r_k}, r_c={args.r_c}），"
          f"点簇 {b_p} scalars/簇 ⇒ 比值 {b_g/b_p:.2f}×\n"
          f"       W_gauss={args.W_gauss} ⇒ equal-bytes 的 W_point={W_point}", flush=True)

    RetainCache.prepare = _patched_prepare
    m = ModelKVzip(args.model, kv_type="retain", gate_path_or_name=args.gate)
    L, H = m.config.num_hidden_layers, m.config.num_key_value_heads
    _ST["d"] = getattr(m.config, "head_dim",
                       m.config.hidden_size // m.config.num_attention_heads)
    for l in range(L):
        m.model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_hook(l))

    ds = load_dataset_all(args.data, m.tokenizer)
    dw = DataWrapper(args.data, ds, m)
    res = defaultdict(list)

    for si in range(args.n_samples):
        qs = [m.apply_template(get_query("qa", q))
              for q in list(ds[si]["question"])[: args.n_queries]]
        kv_f = dw.prefill_context(si, do_score=False)
        S_f = kv_f.key_cache[0].shape[2]
        lg_full = []
        for qi in qs:
            lg_full.append(m.model(qi.to(m.device),
                                   past_key_values=kv_f).logits[0].float().cpu())
            kv_f.slice(S_f)
        del kv_f; torch.cuda.empty_cache()

        kv_p = dw.prefill_context(si, prefill_chunk=args.chunk,
                                  window_size=args.window,
                                  chunk_ratio=args.ratio, level=args.level)
        S_p = kv_p.key_cache[0].shape[2]
        _ST["kv"] = kv_p
        print(f"样本{si}: 建摘要…", flush=True)
        _ST["sum"] = build_summaries(kv_p, L, H,
                                    {"gauss": args.W_gauss, "point": W_point}, args.r_c, args.r_k)
        nk = sum(len(v["g"]["ub"]) for v in _ST["sum"].values() if v)
        nk_p = sum(len(v["p"]["ub"]) for v in _ST["sum"].values() if v)
        print(f"  高斯簇总数 {nk}（{nk*b_g/1e6:.2f}M scalars），"
              f"点簇总数 {nk_p}（{nk_p*b_p/1e6:.2f}M scalars）", flush=True)

        for qi, (qids, lgf) in enumerate(zip(qs, lg_full)):
            lgf_d = lgf.to(m.device)
            for arm in args.arms:
                _ST["arm"] = arm; _ST["on"] = (arm != "none")
                _QCAP.clear()
                lgp = m.model(qids.to(m.device), past_key_values=kv_p).logits[0].float()
                Bv = torch.nn.functional.kl_div(
                    torch.log_softmax(lgp, -1), torch.log_softmax(lgf_d, -1),
                    reduction="none", log_target=True).sum(-1)
                res[arm].append((float(Bv[-1]), float(Bv.mean()), float(Bv.max())))
                _ST["on"] = False
                kv_p.slice(S_p)
                del lgp; torch.cuda.empty_cache()
            print(f"  样本{si} 问题{qi}: " + "  ".join(
                f"{a}={res[a][-1][0]:.3f}" for a in args.arms), flush=True)
        del kv_p; torch.cuda.empty_cache()

    print("\n" + "=" * 92)
    base = np.array([r[0] for r in res["none"]])
    bm = np.array([r[1] for r in res["none"]])
    e0 = np.array([r[0] for r in res["E0"]]) if "E0" in res else None
    ceil_ = (e0 - base).mean() / max(base.mean(), 1e-12) * 100 if e0 is not None else None
    print(f"{'arm':>5} {'n':>4} {'B[last]':>9} {'B[mean]':>9} {'vs none':>9} "
          f"{'vs none(mean)':>14} {'占 oracle':>10}")
    for a in args.arms:
        v = np.array(res[a])
        r1 = (v[:, 0] - base).mean() / max(base.mean(), 1e-12) * 100
        r2 = (v[:, 1] - bm).mean() / max(bm.mean(), 1e-12) * 100
        frac = f"{r1/ceil_*100:9.1f}%" if ceil_ and abs(ceil_) > 1e-9 else "        —"
        print(f"{a:>5} {len(v):>4} {v[:,0].mean():>9.4f} {v[:,1].mean():>9.4f} "
              f"{r1:>8.1f}% {r2:>13.1f}% {frac}")
    print("=" * 92)
    print("判读（H1）：equal-bytes 下 E4（高斯）> E1b（更多点原型） ⇒ 协方差值得那些比特；")
    print("            反之 ⇒ 正确答案是 ResKV 式更多簇，分布式前提在公平记账下失败。")
    print("            E4c − E4 = 一个逐簇 log 校正标量最多能买到多少。")


if __name__ == "__main__":
    main()
