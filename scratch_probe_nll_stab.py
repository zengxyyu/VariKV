#!/usr/bin/env python3
"""效用标签的**信度曲线** —— 单条 KV 的边际效用在多局部的扰动下才是可复现的。

--------------------------------------------------------------------------------
首版（2026-08-16 上午）的结论已撤回，因为对照的量纲错了。记在这里免得重犯：

    pk = pidx[randperm(len(pidx))[:int(len(pidx) * 0.01)]]   # pidx = **全体**可驱逐格子
    V.view(-1)[pk] = ~V.view(-1)[pk]                         # 直接 toggle

`level="pair"` 下全体可驱逐格子 ≈ 28×4×165k = 18.5M，而保留集只有 ~1.44M
（scbench_kv @0.1 的**有效** chunk_ratio 是 0.078，不是 0.1）。于是"1% 扰动"实际是

    翻掉 185k 格 = 保留集的 **12.8%**；
    被翻的 ~90% 原本是驱逐态 ⇒ 预算从 1.44M 涨到 1.60M，**+10.8%**。

所以首版比较的根本不是"两个邻近的等预算存活集合"，而是"10% 预算的缓存"对
"约 10.8% 预算、且随机塞进 17 万条低分 token 的缓存"。测到的 ρ=−0.22 只能说明
**单条效用对缓存构型敏感**，不能说明等预算局部扰动下没有稳定成分。

同一个量纲错误在对照 B 上也犯过（`G=256` 只占保留集 0.0139%）。**任何"扰动多少 /
删掉多少"都要按保留集 `|S|` 取比例。**

--------------------------------------------------------------------------------
本版的三个改动

1. **严格等预算互换**：从 `S∖cand` 抽 `n_swap` 条踢出，同时从 `S̄∖cand` 抽 `n_swap`
   条放进，`|S'| = |S|` 逐位成立（断言）。
2. **ε 相对 `|S|` 定义**，并扫多档 —— 输出的是一条**信度曲线**而不是单点。
   曲线形状本身就是答案：
       ε=0.1% 就 ρ≈0      ⇒ 单条边际确实不是良定义的量（首版想说的那件事，这才算证明）
       ρ 随 ε 单调衰减     ⇒ 效用局部稳定、全局 set-dependent；教师可用，但标签要在
                            与训练时相同的 S 分布上取，且噪声决定所需数据量（∝1/ρ）
3. **同一批扰动下同时测 `U^attn` 的信度。** 这是最有决策价值的一格，而且几乎免费
   （闭式秩一更新，不用前向）。被训练的打分器用的标签是 `U^attn` 而不是 `U^NLL`，
   所以只有它的信度才直接约束"教师能不能学"：
       `U^attn` 信度高而 `U^NLL` 低 ⇒ 教师标签自洽，但它代理的东西不自洽（靶子问题）
       两个都低                     ⇒ 教师标签自身就抖，`ratio × 样本` 再多也没用
       两个都高                     ⇒ 首版结论彻底反了，回到"靶子是否错位"的原问题

另外两档扰动分布，因为它们对应两种不同的问题：
   `--swap_mode random`   全局随机互换 —— 测 set sensitivity 的上界
   `--swap_mode boundary` 只在阈值邻域互换 —— 真实 reranker 只会改动决策边界附近的
                          成员，这一档才是 v2 残差实际所处的工作点
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
_P = os.path.join(ROOT, "external/FastKVzip/prefill")
sys.path.insert(0, _P)
os.chdir(_P)

from data import DataWrapper, load_dataset_all                  # noqa: E402
from model import ModelKVzip                                    # noqa: E402
from utils import set_gen_length                                # noqa: E402


@torch.inference_mode()
def nll(model, ids, n_ans, kv):
    p = model._prob(ids, kv)[-n_ans - 1:-1].float()
    lab = ids[0, -n_ans:]
    return float(-p.gather(1, lab[:, None]).clamp_min(1e-12).log().mean())


@torch.inference_mode()
def attn_util(model, kv, qcap, cand, keep_flat, n_ctx, sink, H, N):
    """`U^attn`：给定存活集合 S 的边际效用，`err(o)=‖W_O(o_full−o_S)‖²`。

    与 `scratch_ctrl_teacher.py:utility_setmarginal` 同式（softmax 的秩一更新）。
    **按 (层, kv头) 分组**后再循环候选 —— `e`/`o_full`/`Gram` 只依赖 (l,h) 不依赖
    候选，也不依赖扰动，逐候选重算会把 [G,T,169k] 的指数表建上百遍。
    """
    d = model.config.hidden_size // model.config.num_attention_heads
    grp = {}
    for t, (l, h, i) in enumerate(cand):
        grp.setdefault((l, h), []).append((t, i))
    out = np.zeros(len(cand))
    for (l, h), items in grp.items():
        Aq = qcap[l][0].float() * (d ** -0.5)
        HQ, T, _ = Aq.shape
        G = HQ // H
        Aq = Aq.view(H, G, T, d)[h]
        K = kv.key_cache[l][0][h, :n_ctx].float()
        Vv = kv.value_cache[l][0][h, :n_ctx].float()
        WO = model.model.model.layers[l].self_attn.o_proj.weight.detach().float()
        W = WO[:, h * G * d:(h + 1) * G * d]
        Gram = W.T @ W
        a = torch.einsum("gtd,nd->gtn", Aq, K)
        e = (a - a.amax(-1, keepdim=True)).exp()
        o_full = torch.einsum("gtn,nd->gtd", e, Vv) / e.sum(-1, keepdim=True)
        # off-by-sink：`valid` 只覆盖可驱逐区，前 sink 个永远保留，必须补 True 对齐
        mv = keep_flat.view(-1, H, N)[l, h].to(K.device)
        m = torch.cat([torch.ones(sink, dtype=mv.dtype, device=K.device), mv])
        eS = e * m[None, None, :]
        ZS = eS.sum(-1, keepdim=True).clamp_min(1e-30)
        NS = torch.einsum("gtn,nd->gtd", eS, Vv)

        def err(o):
            z = (o_full - o).permute(1, 0, 2).reshape(T, G * d)
            return ((z @ Gram) * z).sum(-1).mean()

        es = err(NS / ZS)
        for (t, i) in items:
            ia = i + sink
            sg = -1.0 if bool(m[ia]) else 1.0
            Zp = (ZS + sg * e[..., ia:ia + 1]).clamp_min(1e-30)
            Np = NS + sg * e[..., ia:ia + 1] * Vv[ia]
            eo = err(Np / Zp)
            out[t] = float(eo - es) if bool(m[ia]) else float(es - eo)
        del a, e, eS, NS
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="scbench_kv")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--n_cand", type=int, default=20)
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0.001, 0.005, 0.02, 0.10],
                    help="互换比例，**相对保留集 |S|**，不是相对全体格子")
    ap.add_argument("--swap_mode", default="random", choices=["random", "boundary"])
    ap.add_argument("--no_attn", action="store_true", help="跳过 U^attn（省一次满缓存预填）")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--chunk", type=int, default=16000)
    ap.add_argument("--window", type=int, default=4096)
    a = ap.parse_args()

    m = ModelKVzip(a.model, "retain", "fastkvzip")
    ds = DataWrapper(a.data, load_dataset_all(a.data, m.tokenizer), m)
    set_gen_length(a.data, m)
    L = m.config.num_hidden_layers
    g = torch.Generator(device="cpu").manual_seed(0)
    det, U_nll, U_att = [], {}, {}          # eps -> list of per-sample arrays

    for si in range(min(a.num, len(ds))):
        qcap, ids, n_ans, n_ctx = None, None, None, None
        kv_f = ds.prefill_context(si, do_score=False)
        inputs, _ = ds.generate_answer(si, kv_f, prob=False)
        task = "qa" if "qa" in inputs else list(inputs.keys())[0]
        ids = torch.cat([inputs[task][k] for k in ("q", "a")], dim=-1)
        n_ans = len(inputs[task]["a"][0])
        n_ctx = kv_f.key_cache[0].shape[2]
        if not a.no_attn:
            kv_f.capture_q, kv_f._q_cap = True, {}
            m.model(ids, past_key_values=kv_f)
            kv_f.capture_q = False
            qcap = {l: kv_f._q_cap[l] for l in range(L)}
        del kv_f
        torch.cuda.empty_cache()

        kv = ds.prefill_context(si, prefill_chunk=a.chunk, window_size=a.window,
                                chunk_ratio=a.ratio, level="pair")
        kv.valid = kv.valid.clone()     # prefill 在 inference_mode 下建的张量不可就地改
        V, sink = kv.valid, kv.sink
        H, N = V.shape[1], V.shape[2]
        Vf = V.view(-1)
        base = nll(m, ids, n_ans, kv)
        det.append(abs(nll(m, ids, n_ans, kv) - base))      # A：确定性（应恒为 0）

        sc = torch.stack(kv.score, 0)[:, 0]
        s_flat = sc[..., sink:sink + N].float().reshape(-1)
        tau = s_flat.sort(descending=True).values[
            max(int(s_flat.numel() * a.ratio) - 1, 0)]
        cand_i = (s_flat - tau).abs().argsort()[:a.n_cand]
        cand = [(int(t // (H * N)), int((t // N) % H), int(t % N))
                for t in cand_i.tolist()]
        n_ret0 = int(Vf.sum())
        print(f"  样本 {si}: base {base:.4f}  |S| {n_ret0/1e6:.3f}M / "
              f"{len(s_flat)/1e6:.2f}M 格 (有效 ratio {n_ret0/len(s_flat):.4f})", flush=True)

        # 候选排除在互换池外：否则测的是"翻它两次"而不是"在不同背景下翻它"
        free = torch.ones(len(s_flat), dtype=torch.bool)
        free[cand_i] = False
        ret_pool = (Vf.cpu() & free).nonzero(as_tuple=True)[0]
        evi_pool = (~Vf.cpu() & free).nonzero(as_tuple=True)[0]
        if a.swap_mode == "boundary":
            # 真实 reranker 只动决策边界附近的成员，按 |s−τ| 升序取
            dist = (s_flat - tau).abs().cpu()
            ret_pool = ret_pool[dist[ret_pool].argsort()]
            evi_pool = evi_pool[dist[evi_pool].argsort()]

        def measure():
            b = nll(m, ids, n_ans, kv)
            un = []
            for (l, h, i) in cand:
                t = l * H * N + h * N + i
                kept = bool(Vf[t])
                Vf[t] = not kept
                n2 = nll(m, ids, n_ans, kv)
                Vf[t] = kept
                un.append((n2 - b) if kept else (b - n2))
            ua = (None if a.no_attn else
                  attn_util(m, kv, qcap, cand, Vf, n_ctx, sink, H, N))
            return np.array(un), ua

        for eps in [0.0] + list(a.eps):
            if eps > 0:
                ns = max(int(n_ret0 * eps), 1)
                if ns > min(len(ret_pool), len(evi_pool)):
                    continue
                out_i = (ret_pool[torch.randperm(len(ret_pool), generator=g)[:ns]]
                         if a.swap_mode == "random" else ret_pool[:ns])
                in_i = (evi_pool[torch.randperm(len(evi_pool), generator=g)[:ns]]
                        if a.swap_mode == "random" else evi_pool[:ns])
                Vf[out_i.to(Vf.device)] = False
                Vf[in_i.to(Vf.device)] = True
                # **等预算是这个探针的全部意义所在**，所以断言而不是相信构造
                assert int(Vf.sum()) == n_ret0, (int(Vf.sum()), n_ret0)
            un, ua = measure()
            U_nll.setdefault(eps, []).append(un)
            if ua is not None:
                U_att.setdefault(eps, []).append(ua)
            if eps > 0:
                Vf[out_i.to(Vf.device)] = True
                Vf[in_i.to(Vf.device)] = False
                assert int(Vf.sum()) == n_ret0
        from scipy.stats import spearmanr
        msg = "  ".join(
            f"ε={e:g}:{spearmanr(U_nll[0.0][-1], U_nll[e][-1]).statistic:+.2f}"
            for e in a.eps if e in U_nll)
        print(f"    ρ(U^NLL(S), U^NLL(S_ε))  {msg}", flush=True)
        del kv
        torch.cuda.empty_cache()

    from scipy.stats import spearmanr
    np.savez(os.path.join(ROOT, f"scratch_nllstab_{a.data}_{a.swap_mode}.npz"),
             **{f"nll_{e}": np.array(v) for e, v in U_nll.items()},
             **{f"att_{e}": np.array(v) for e, v in U_att.items()})
    n = len(U_nll[0.0])
    print(f"\n=== {a.data} @ ratio {a.ratio}　{n} 篇 × {a.n_cand} 候选　"
          f"互换模式 {a.swap_mode} ===")
    print(f"A 确定性：同掩码两次 NLL |Δ| 最大 {max(det):.3e}（应为 0）")
    print(f"\n{'ε (相对|S|)':>12}{'ρ(U^NLL)':>12}{'逐样本中位':>12}"
          f"{'ρ(U^attn)':>12}{'逐样本中位':>12}")
    for e in a.eps:
        if e not in U_nll:
            continue
        def rho(D):
            cat = spearmanr(np.concatenate(D[0.0]), np.concatenate(D[e])).statistic
            per = [spearmanr(x, y).statistic for x, y in zip(D[0.0], D[e])]
            return cat, float(np.median(per))
        rn, mn = rho(U_nll)
        ra, ma = (rho(U_att) if e in U_att else (float("nan"),) * 2)
        print(f"{e:>12g}{rn:>12.3f}{mn:>12.3f}{ra:>12.3f}{ma:>12.3f}")
    print("\n判读：")
    print("  最小 ε 就 ρ(U^NLL)≈0        ⇒ 单条边际不是良定义的量（首版想证的那件事）")
    print("  ρ 随 ε 单调衰减              ⇒ 局部稳定、全局 set-dependent；教师可用，"
          "但所需数据量 ∝ 1/信度")
    print("  ρ(U^attn) 高而 ρ(U^NLL) 低   ⇒ 教师标签自洽，它代理的东西不自洽（靶子问题）")
    print("  两个都低                     ⇒ 教师标签自身就抖，加样本也救不回来")


if __name__ == "__main__":
    raise SystemExit(main())
