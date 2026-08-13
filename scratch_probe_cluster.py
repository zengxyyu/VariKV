"""分簇 × (质量, 方向) × oracle 的完整误差分解 —— 决定还有没有方法可做

**为什么必须重做。** 我们的 `centroid.py:183` 用 `b = pos // W`（纯位置分桶），而
**ResKV（2607.29591）Eq 12 用 k-means（Lloyd）在 key 空间分簇**，正是为了满足它
Eq 11 写出的成立条件 "if the tokens in C_j have similar logits for a query"。
P0 §5.4 测到"位置局部性根本不减少打分离散度"，我曾把它当成原理性限制 —— 它其实
是对**我们自己实现选择**的批评。所以先前测到的「质量低估 13.9×」与由它推出的
γ≈0.75 收缩故事，都可能是位置分桶的产物，必须在 k-means 下重测。

**但 k-means 不保证缺口消失，这一点要先算清楚再看数。**
    log J(q) ≈ ½·aᵀΣ_C·a,   a = q/√d,  Σ_C = 簇内 key 协方差
k-means 最小化的是 Σ_j n_j·tr(Σ_{C_j})，即**迹** —— 各向同性 query 下的平均。
真实 query 高度非各向同性，若落在 Σ_C 大特征值的子空间，aᵀΣ_C a 可远大于 tr(Σ)/d。
实测 log(D/D̂)=2.63 ⇒ Var(aᵀδ)≈5.26（对上 P0 的 5.14）；要压到 1× 需降到 ~0.1，
即 **50 倍**。K=16 在 R^128 里对 15 万 token 只分 16 簇，不可能做到。
**预测：k-means 显著减小缺口但远不到 1×。**

**key 空间分簇只帮质量，不帮方向。** ResKV Eq 11 要两个近似同时成立：
    Σ e^{a_p} ≈ c_j e^{ā_j}          ← k-means 优化的是这个
    Σ e^{a_p} v_p ≈ c_j e^{ā_j} v̄_j  ← 键相近 ≠ 值相近，k-means 帮不上
AM 用 OLS 拟合 C_v 就是在处理后者。而 massdir 探针已测到 e=0.676 却 r=0.993
（范数几乎精确、误差全在方向），一旦质量修好，方向就成为绑定约束。

**输出矩阵**，每个分簇 ∈ {position, k-means, random} 各一行：

| 量 | 含义 |
|---|---|
| `log D̂/D` 0 阶 | Jensen 缺口（`log n_j` 版，即 ResKV） |
| `log D̂/D` 2 阶 | 加 ½Var_j(aᵀδ) 后 —— **候选的零拟合修法** |
| `cos(v̂,v)` / `r` / `e` | 方向估计量质量 |
| 四格 oracle | (D̂,v̂) 现方法 / (D,v̂) 只修质量 / (D̂,v) 只修方向 / (D,v)=满缓存自检 |
| γ-sweep | 最优信任度；问「k-means 之后 γ* 还小于 1 吗」 |

**random 分簇是关键对照**：如果它和 k-means 差不多，说明分簇策略根本不是主变量
（stage-1 里"随机驱逐打败所有原则性准则"是同一类教训）。

判读（预注册，2×2）：

| k-means 质量缺口 | k-means γ* | 含义 |
|---|---|---|
| 小 | ≈1 | 旧的 centroid/Jensen/shrinkage 故事基本死 |
| 小 | <1 | 质量不是问题，方向/可靠性可能才是 |
| 大 | ≈1 | 质量修正有价值，但 AM 的拟合 β 已覆盖 ⇒ 必须有零拟合优势 |
| **大** | **<1** | **最有意思：既缺质量、又不能盲目补满 ⇒ 存在真正的校准问题** |

注意「缺口大且 γ*<1」仍然意味着**净放大**（实测 0.75/0.072 ≈ 10.4×），只是不放大到真值 ——
这正是收缩估计量该做的事：瞄准 γ*·真值，而不是原始低估值。

**不改 `centroid.py`** —— 分簇在探针内部离线重算，避免把 harness、mask、cache state
一起改动引入新的混淆，也避免影响正在跑的评估任务。

用法：CUDA_VISIBLE_DEVICES=k .venv/bin/python scratch_probe_cluster.py \
          --data scbench_kv --K 16 --n 0
"""
import argparse
import sys
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

GAMMAS = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
SCHEMES = ("position", "kmeans", "random")
_Q = {}
_orig_prepare = RetainCache.prepare


def _patched_prepare(self, q, k, v, l):
    _Q[l] = q.detach().clone()
    return _orig_prepare(self, q, k, v, l)


def cs(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def assign(scheme, k_ev, pos, K, W, iters=4, gen=None):
    """→ [n] 的簇标签。position 复刻 centroid.py:183；kmeans 复刻 ResKV Eq 12。"""
    n = k_ev.shape[0]
    if scheme == "position":
        return (pos // W).clamp(max=K - 1)
    if scheme == "random":
        return torch.randint(0, K, (n,), device=k_ev.device, generator=gen)
    # k-means（Lloyd），**从高分 token 初始化** —— ResKV 说 "initialized from
    # high-scored tokens in E"；这里没有分数，用等距抽样代替（同样是确定性的）。
    idx = torch.linspace(0, n - 1, K).long().to(k_ev.device)
    C = k_ev[idx].clone()
    kn = (k_ev * k_ev).sum(-1, keepdim=True)              # [n,1]
    for _ in range(iters):
        # 分块算距离，避免 [n,K] 在 K=1024 时过大
        lab = torch.empty(n, dtype=torch.long, device=k_ev.device)
        step = max(1, int(2 ** 24 // max(K, 1)))
        for s in range(0, n, step):
            e = min(n, s + step)
            d2 = kn[s:e] - 2.0 * (k_ev[s:e] @ C.T) + (C * C).sum(-1)[None]
            lab[s:e] = d2.argmin(-1)
        cnt = torch.zeros(K, device=k_ev.device).index_add_(
            0, lab, torch.ones(n, device=k_ev.device))
        Cn = torch.zeros_like(C).index_add_(0, lab, k_ev)
        keep = cnt > 0
        C = torch.where(keep[:, None], Cn / cnt.clamp_min(1.0)[:, None], C)
    return lab


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="scbench_kv")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(0, 28, 3)))
    ap.add_argument("--mem_frac", type=float, default=0.0)
    a = ap.parse_args()

    RetainCache.prepare = _patched_prepare
    if a.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    m = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M", kv_type="centroid",
                   gate_path_or_name="fastkvzip")
    m.varikv_K = a.K
    m.varikv_rope_mode = "post"
    H = m.config.num_key_value_heads
    d = getattr(m.config, "head_dim", m.config.hidden_size // m.config.num_attention_heads)
    gen = torch.Generator(device=m.device); gen.manual_seed(0)

    ds = load_dataset_all(a.data, m.tokenizer)
    dw = DataWrapper(a.data, ds, m)
    if a.n <= 0:
        a.n = len(ds) - a.start
    print(f"[cfg] {a.data} K={a.K} ratio={a.ratio} 层 {len(a.layers)} 个 "
          f"样本 {a.n} 条 H={H} d={d}", flush=True)

    rows = []          # (scheme_id, 各项指标…)
    gam = []           # (scheme_id, γ 曲线…)
    for i in range(a.start, a.start + a.n):
        kv = dw.prefill_context(i, prefill_chunk=16000, window_size=4096,
                                chunk_ratio=a.ratio, level="pair")
        q_ids = m.apply_template(get_query("qa", list(ds[i]["question"])[0])).to(m.device)
        _Q.clear()
        m.model(q_ids, past_key_values=kv)
        S = kv.key_cache[0].shape[2]
        for l in a.layers:
            WO = m.model.model.layers[l].self_attn.o_proj.weight.detach().float()
            v_ = kv._get_valid(l, S)
            while v_.dim() > 2:
                v_ = v_.squeeze(0)
            valid = v_.bool().to(m.device)
            kall = kv.key_cache[l][0].float(); vall = kv.value_cache[l][0].float()
            T = _Q[l].shape[2]; G = _Q[l].shape[1] // H
            for h in range(H):
                ev = (~valid[h]).nonzero(as_tuple=True)[0]
                if ev.numel() < max(64, 4 * a.K):
                    continue
                k_ev, v_ev = kall[h, ev], vall[h, ev]
                k_rt, v_rt = kall[h, valid[h]], vall[h, valid[h]]
                labs = {s: assign(s, k_ev, ev, a.K, kv.W, gen=gen) for s in SCHEMES}
                for g in range(G):
                    hq = h * G + g
                    W = WO[:, hq * d:(hq + 1) * d]
                    aq = _Q[l][0].view(H, G, T, d)[h, g, -1].float() * (d ** -0.5)
                    sR, sE = aq @ k_rt.T, aq @ k_ev.T
                    LR, LE = torch.logsumexp(sR, -1), torch.logsumexp(sE, -1)
                    oR = torch.softmax(sR, -1) @ v_rt
                    vE = torch.softmax(sE, -1) @ v_ev          # 真方向
                    def cell(LEx, vx):
                        lam = torch.exp(LR - torch.logaddexp(LR, LEx))
                        return lam * oR + (1 - lam) * vx
                    o_full = cell(LE, vE)
                    dfull = o_full - oR
                    nd = (W @ dfull).norm().clamp_min(1e-12)
                    for si, s in enumerate(SCHEMES):
                        lab = labs[s]
                        z1 = torch.zeros(a.K, device=aq.device)
                        cnt = z1.clone().index_add_(0, lab, torch.ones_like(sE))
                        cl = cnt.clamp_min(1.0)
                        kbar = torch.zeros(a.K, d, device=aq.device).index_add_(
                            0, lab, k_ev) / cl[:, None]
                        vbar = torch.zeros(a.K, d, device=aq.device).index_add_(
                            0, lab, v_ev) / cl[:, None]
                        occ = cnt > 0
                        logn = torch.where(occ, cl.log(), torch.full_like(cl, -1e30))
                        sbar = aq @ kbar.T
                        # 0 阶（= ResKV）与 2 阶（候选零拟合修法）的质量
                        delta = sE - sbar[lab]
                        var_ = z1.clone().index_add_(0, lab, delta * delta) / cl
                        r0 = sbar + logn
                        r2 = sbar + torch.where(occ, logn + 0.5 * var_, logn)
                        LE0, LE2 = torch.logsumexp(r0, -1), torch.logsumexp(r2, -1)
                        vh = torch.softmax(r0, -1) @ vbar
                        e_ = float((vh - vE).norm() / vE.norm().clamp_min(1e-30))
                        rows.append((si,
                                     float(LE0 - LE), float(LE2 - LE),
                                     e_, float(vh.norm() / vE.norm().clamp_min(1e-30)),
                                     cs(vh, vE),
                                     float((W @ (cell(LE0, vh) - o_full)).norm() / nd),
                                     float((W @ (cell(LE, vh) - o_full)).norm() / nd),
                                     float((W @ (cell(LE0, vE) - o_full)).norm() / nd),
                                     float((W @ (cell(LE2, vh) - o_full)).norm() / nd),
                                     float(var_[occ].mean()) if occ.any() else 0.0,
                                     1.0 / (1.0 + e_ * e_)))
                        errs = [si]
                        for gm in GAMMAS:
                            LEg = (LE + np.log(gm)) if gm > 0 else torch.tensor(
                                -1e30, device=LE.device, dtype=LE.dtype)
                            errs.append(float((W @ (cell(LEg, vh) - o_full)).norm() / nd))
                        gam.append(errs)
        del kv
        torch.cuda.empty_cache()
        print(f"  样本 {i} 完成，累计 {len(rows)} 行", flush=True)

    A = np.array(rows); Gm = np.array(gam)
    print("\n" + "=" * 112)
    print(f"分簇 × (质量,方向) × oracle 分解　{a.data} @ratio {a.ratio}　K={a.K}　"
          f"{a.n} 条样本　{len(A)//len(SCHEMES)} 个 (层,查询头)/分簇")
    print("-" * 112)
    hdr = (f"{'分簇':<10}{'logD̂/D 0阶':>13}{'logD̂/D 2阶':>13}{'Var(aᵀδ)':>11}"
           f"{'cos(v̂,v)':>11}{'‖v̂‖/‖v‖':>11}{'e':>8}")
    print(hdr)
    for si, s in enumerate(SCHEMES):
        B = A[A[:, 0] == si]
        print(f"{s:<10}{np.median(B[:,1]):>+13.3f}{np.median(B[:,2]):>+13.3f}"
              f"{np.median(B[:,10]):>11.2f}{np.median(B[:,5]):>11.4f}"
              f"{np.median(B[:,4]):>11.4f}{np.median(B[:,3]):>8.4f}")
    print("-" * 112)
    print(f"{'分簇':<10}{'现方法':>11}{'只修质量':>11}{'只修方向':>11}{'2阶质量':>11}"
          f"　（‖o−o_full‖/‖Δo‖ 中位，1.0=完全不修）")
    for si, s in enumerate(SCHEMES):
        B = A[A[:, 0] == si]
        print(f"{s:<10}{np.median(B[:,6]):>11.4f}{np.median(B[:,7]):>11.4f}"
              f"{np.median(B[:,8]):>11.4f}{np.median(B[:,9]):>11.4f}")
    print("-" * 112)
    print("γ-sweep（保持各自的 v̂，只缩放真实质量 γ·D_E）")
    print(f"{'分簇':<10}" + "".join(f"{g:>9.2f}" for g in GAMMAS) + f"{'  最优 γ':>10}")
    for si, s in enumerate(SCHEMES):
        B = Gm[Gm[:, 0] == si][:, 1:]
        med = [float(np.median(B[:, j])) for j in range(len(GAMMAS))]
        print(f"{s:<10}" + "".join(f"{v:>9.4f}" for v in med)
              + f"{GAMMAS[int(np.argmin(med))]:>10g}")
    print("=" * 112)
    print("判读（预注册）：")
    print("  k-means 缺口小 且 γ*≈1  ⇒ 旧的 centroid/Jensen/shrinkage 故事基本死")
    print("  k-means 缺口小 但 γ*<1  ⇒ 质量不是问题，方向/可靠性才是")
    print("  k-means 缺口大 且 γ*≈1  ⇒ 质量修正有价值，但 AM 的拟合 β 已覆盖 ⇒ 需零拟合优势")
    print("  k-means 缺口大 且 γ*<1  ⇒ **既缺质量又不能盲目补满**，存在真正的校准问题")
    print("  random ≈ k-means        ⇒ 分簇策略根本不是主变量（对照，别漏看这一行）")
    print("  「2阶质量」若把误差压到接近「只修质量」⇒ 零拟合闭式修法可行，这是候选方法")
    print("注意：本探针只测局部注意力误差。Retr.MultiHop 已证明「更接近满缓存」≠")
    print("     「任务分数更高」，所以任何 γ / 修法都还需要一次下游验证。")


if __name__ == "__main__":
    main()
