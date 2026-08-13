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
SCHEMES = ("position", "random", "kmeans", "Cq-kmeans", "score-oracle")
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
    if scheme in ("score-oracle", "Cq-kmeans"):
        raise RuntimeError(f"{scheme} 需在 g 循环内单独构造")
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


def report(A, Gm, LAY, a, deff_cq, nd_):
    """完整表。中途与结尾共用同一份代码，避免两处口径分叉。"""
    md = lambda B, c: float(np.median(B[:, c]))                  # noqa: E731
    print("\n" + "=" * 118)
    print(f"分簇 × (质量,方向) × oracle 分解　{a.data} @ratio {a.ratio}　K={a.K}　"
          f"{nd_} 条样本　{len(A)//len(SCHEMES)} 个 (层,查询头)/分簇")
    print("-" * 118)
    dq = [v for v in deff_cq.values() if isinstance(v, tuple)]
    if dq:
        print(f"C_q = E[aaᵀ]（**二阶矩，非协方差**）d_eff 中位 "
              f"{np.median([x[0] for x in dq]):.1f}　"
              f"top-3 累积谱 {np.median([x[1] for x in dq]):.3f}　"
              f"top-8 {np.median([x[2] for x in dq]):.3f}")
        print("  ⇒ 低秩截断可行性看**累积谱**，不看 d_eff；成本是 O(ndr+nKrI)，"
              "不是 O(nKr)")
    print(f"被驱逐 key 的**有效维度** d_eff（参与率）中位 {md(A, 13):.1f}"
          f"　⇒ 量化失真下界 ~K^(−2/d_eff) = {a.K ** (-2.0 / max(md(A,13),1e-9)):.4f}"
          f"（即 K={a.K} 最多把簇内方差降到这个比例）")
    print("-" * 118)
    print(f"{'分簇':<14}{'logD̂/D 0阶':>12}{'2阶oracle':>11}{'2阶1标量':>11}"
          f"{'簇内/总 方差':>13}{'cos(v̂,v)':>10}{'‖v̂‖/‖v‖':>10}{'e':>8}")
    for si, s in enumerate(SCHEMES):
        B = A[A[:, 0] == si]
        if not len(B):
            continue
        print(f"{s:<14}{md(B,1):>+12.3f}{md(B,2):>+11.3f}{md(B,11):>+11.3f}"
              f"{md(B,12):>13.4f}{md(B,5):>10.4f}{md(B,4):>10.4f}{md(B,3):>8.4f}")
    print("-" * 118)
    print("四格 oracle + 二阶（‖o−o_full‖/‖Δo‖ 中位，1.0 = 完全不修正）")
    print(f"{'分簇':<14}{'现方法':>10}{'只修质量':>10}{'只修方向':>10}"
          f"{'交互':>9}{'2阶oracle':>11}{'2阶1标量':>11}")
    for si, s in enumerate(SCHEMES):
        B = A[A[:, 0] == si]
        if not len(B):
            continue
        # L(D,v)=0（恒等）⇒ 交互 = 0 − mc − cd + cc。**必须报交互**：
        # 质量与方向是强交互变量（N_E = D_E·v_E），不能只看两个主效应排瓶颈。
        inter = md(B, 6) - md(B, 7) - md(B, 8)
        print(f"{s:<14}{md(B,6):>10.4f}{md(B,7):>10.4f}{md(B,8):>10.4f}"
              f"{inter:>+9.4f}{md(B,9):>11.4f}{md(B,10):>11.4f}")
    print("-" * 118)
    print("γ-sweep（保持各自的 v̂，只缩放真实质量 γ·D_E）")
    print(f"{'分簇':<14}" + "".join(f"{g:>9.2f}" for g in GAMMAS) + f"{'  最优 γ':>10}")
    for si, s in enumerate(SCHEMES):
        B = Gm[Gm[:, 0] == si][:, 1:]
        if not len(B):
            continue
        med = [float(np.median(B[:, k])) for k in range(len(GAMMAS))]
        print(f"{s:<14}" + "".join(f"{v:>9.4f}" for v in med)
              + f"{GAMMAS[int(np.argmin(med))]:>10g}")
    if len(LAY):
        print("-" * 118)
        print("**层级聚合误差**（‖Σ_h W_h(ô−o_full)‖ / ‖Σ_h W_h Δo‖）—— 这才是残差流看到的")
        print(f"{'分簇':<14}{'现方法':>10}{'只修质量':>10}{'只修方向':>10}"
              f"{'2阶oracle':>11}{'2阶1标量':>11}{'cos(现方法)':>12}")
        for si, s in enumerate(SCHEMES):
            B = LAY[LAY[:, 0] == si]
            if not len(B):
                continue
            print(f"{s:<14}{np.median(B[:,2]):>10.4f}{np.median(B[:,3]):>10.4f}"
                  f"{np.median(B[:,4]):>10.4f}{np.median(B[:,5]):>11.4f}"
                  f"{np.median(B[:,6]):>11.4f}{np.median(B[:,7]):>12.4f}")
    print(f"\n  收缩理论预测 γ* = 1/(1+e²) 中位 = {md(A, 14):.4f}")
    print("=" * 118)
    print("判读（预注册）：")
    print("  **score-oracle 是最关键的一行。** 它按 aᵀk_i 在一维上分等量桶 ⇒ 簇内 logit")
    print("     方差按构造最小 ⇒ 质量近乎精确，剩下的输出误差**纯粹是 value 方向误差**。")
    print("     若 score-oracle 缺口≈0 而 k-means 不是 ⇒ 瓶颈是**缺 query 信息**，不是容量，")
    print("     这正好解释 Attention Matching 为何必须用 reference queries。")
    print("     若连 score-oracle 都留大缺口 ⇒ 容量才是绑定约束。")
    print("  k-means 缺口小 且 γ*≈1  ⇒ 旧的 centroid/Jensen/shrinkage 故事基本死")
    print("  k-means 缺口大 且 γ*<1  ⇒ 存在真正的校准问题（注意这仍是净放大 γ*/r_D，")
    print("     是标准 bias–variance 收缩，不是「越准越坏」的反常）")
    print("  random ≈ k-means        ⇒ 分簇策略根本不是主变量（别漏看这一行）")
    print("  「2阶1标量」若接近「2阶oracle」⇒ 各向同性假设成立，每簇 +1 scalar 可部署；")
    print("     若差很远 ⇒ 只有 oracle 版有效，而存 Σ_kk 是 +6376% 状态，压缩故事死")
    print("注意：本探针只测局部注意力误差。Retr.MultiHop 已证明「更接近满缓存」≠")
    print("     「任务分数更高」，任何 γ / 修法都还需要一次下游验证。")




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
    ap.add_argument("--cq_from", default="",
                    help="从**另一个数据集**估 C_q 然后冻结。空 = 用本数据集前序样本"
                         "（transductive，会看到测试任务的 query 几何，只能当诊断）")
    ap.add_argument("--cq_n", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--report_every", type=int, default=10,
                    help="每这么多条样本打一次完整表 + 落盘，便于中途看效果")
    a = ap.parse_args()

    CQ, CQN, deff_cq = {}, {}, {}   # C_q = E[aaᵀ] 的累计（见下面 --cq_from）
    if not a.out:
        a.out = f"scratch_cluster_{a.data}_K{a.K}_s{a.start}"
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

    CQ_FROZEN = False
    if a.cq_from:
        # **held-out C_q**：从另一个任务/模板的 query 估，然后冻结。这是唯一能回答
        # 「Cq 的收益是真可部署，还是来自看过测试 query 分布」的做法。
        ds2 = load_dataset_all(a.cq_from, m.tokenizer)
        dw2 = DataWrapper(a.cq_from, ds2, m)
        print(f"[cq] 从 {a.cq_from} 的 {a.cq_n} 条估 C_q 后冻结", flush=True)
        for j in range(a.cq_n):
            kv2 = dw2.prefill_context(j, prefill_chunk=16000, window_size=4096,
                                      chunk_ratio=a.ratio, level="pair")
            q2 = m.apply_template(get_query("qa", list(ds2[j]["question"])[0])).to(m.device)
            _Q.clear(); m.model(q2, past_key_values=kv2)
            for l in a.layers:
                Aq = _Q[l][0].float() * (d ** -0.5)
                G2 = Aq.shape[0] // H
                for h in range(H):
                    z = Aq.view(H, G2, -1, d)[h].reshape(-1, d)
                    CQ[(l, h)] = CQ.get((l, h), torch.zeros(d, d, device=z.device)) + z.T @ z
                    CQN[(l, h)] = CQN.get((l, h), 0) + z.shape[0]
            del kv2; torch.cuda.empty_cache()
        CQ_FROZEN = True
    rows = []          # (scheme_id, 各项指标…)
    lay_rows = []      # (scheme_id, layer, 层级聚合误差…)
    gam = []           # (scheme_id, γ 曲线…)
    # **C_q = E[aaᵀ] 的正确性来自 E_q[(aᵀ(k−μ))²] = (k−μ)ᵀC_q(k−μ)** —— 它是
    # 「未来 query 分布下的期望平方 logit 误差」这个 distortion 的精确对象，不是启发式。
    # 但它必须**无泄漏**：只用样本 i **之前**的 query 累计，所以是模型级统计量，可部署。
    # 记法：C_q = LᵀL ⇒ (k−μ)ᵀC_q(k−μ) = ‖L(k−μ)‖² ⇒ Mahalanobis k-means
    # 退化成「先做 C_q^{1/2} 变换、再普通 Euclidean k-means」。
    # 复杂度诚实说明：**这不是 O(nd)**。Lloyd 的 assignment 是每轮 O(nKr)，
    # r 是 C_q 的有效秩；只有 r≪d 时才比 Euclidean 的 O(nKd) 便宜。

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
            # **层级聚合。** 逐头误差不是层误差：层输出是 Σ_h W_h Δo_h，而 P0 测过
            # 跨头相消只留 0.21–0.25 ⇒「每头都更准」完全可以「层级更差」。
            # 这一列在旧的 massdir 探针里有，写本探针时被我丢了，属回归。
            AGG = {s: {k: torch.zeros(WO.shape[0], device=m.device)
                       for k in ("cc", "mc", "cd", "2or", "2iso")} for s in SCHEMES}
            AGF = {k: torch.zeros(WO.shape[0], device=m.device) for k in ("full", "R")}
            for h in range(H):
                ev = (~valid[h]).nonzero(as_tuple=True)[0]
                if ev.numel() < max(64, 4 * a.K):
                    continue
                k_ev, v_ev = kall[h, ev], vall[h, ev]
                k_rt, v_rt = kall[h, valid[h]], vall[h, valid[h]]
                labs = {s: assign(s, k_ev, ev, a.K, kv.W, gen=gen)
                        for s in SCHEMES if s not in ("score-oracle", "Cq-kmeans")}
                # **有效维度**（参与率）。高维量化失真下界 ~ K^{-2/d_eff}：d_eff=128 时
                # K=16 只降 4%（16^{-2/128}=0.958），d_eff=8 降 2×，d_eff=4 降 4×，
                # 要降 50× 需 d_eff≈1.4。所以"16 簇不可能大幅压方差"该由 d_eff 判定，
                # 不能拿 ambient 128 维说事。
                kc = k_ev - k_ev.mean(0, keepdim=True)
                Sig = (kc.T @ kc) / kc.shape[0]
                lam = torch.linalg.eigvalsh(Sig.double()).clamp_min(0)
                d_eff = float(lam.sum() ** 2 / (lam * lam).sum().clamp_min(1e-30))
                # C_q 的白化变换（用**之前样本**累计的，无泄漏）
                Cq_ready, dqe = None, 0.0
                if CQN.get((l, h), 0) >= 64:
                    Cqm = (CQ[(l, h)] / CQN[(l, h)]).double()
                    lq, Uq = torch.linalg.eigh(Cqm)
                    lq = lq.clamp_min(0)
                    dqe = float(lq.sum() ** 2 / (lq * lq).sum().clamp_min(1e-30))
                    # **E[aaᵀ] 是二阶矩，不是协方差**：E[aaᵀ]=Σ_q+μμᵀ，强均值会天然造出
                    # 一个 rank-1 成分 ⇒ d_eff 小**不等于**「query 只有那么多维」。
                    # 用于 E[(aᵀδ)²] 的正确矩阵确实是未中心化的 E[aaᵀ]，但解释时必须
                    # 同时看中心化版本，并报累积谱（低秩截断可行性靠它，不靠 d_eff）。
                    lqs = lq.flip(0)
                    cum3 = float(lqs[:3].sum() / lqs.sum().clamp_min(1e-30))
                    cum8 = float(lqs[:8].sum() / lqs.sum().clamp_min(1e-30))
                    deff_cq[(l, h)] = (dqe, cum3, cum8)
                    Lt = (Uq * lq.sqrt()[None]).float()          # k ↦ kᵀU√Λ
                    Cq_ready = lambda x, _L=Lt: x @ _L

                # **query 无关的簇统计只算一次。** kbar/vbar 的 index_add 是
                # 15 万×128 的 scatter，原先在 g 循环里被重算 G=7 次；它们不依赖 q，
                # 提出来后这 4 个分簇各省 7 倍（这才是探针慢的真正原因，不是 eigh）。
                PRE = {}
                for s in SCHEMES:
                    if s == "score-oracle":
                        continue                      # 唯一依赖 q 的，留在 g 循环里
                    if s == "Cq-kmeans":
                        if Cq_ready is None:
                            continue
                        lab = assign("kmeans", Cq_ready(k_ev), ev, a.K, kv.W, gen=gen)
                    else:
                        lab = labs[s]
                    z1 = torch.zeros(a.K, device=k_ev.device)
                    cnt = z1.clone().index_add_(0, lab, torch.ones(
                        k_ev.shape[0], device=k_ev.device))
                    cl = cnt.clamp_min(1.0)
                    kbar = torch.zeros(a.K, d, device=k_ev.device).index_add_(
                        0, lab, k_ev) / cl[:, None]
                    vbar = torch.zeros(a.K, d, device=k_ev.device).index_add_(
                        0, lab, v_ev) / cl[:, None]
                    occ = cnt > 0
                    logn = torch.where(occ, cl.log(), torch.full_like(cl, -1e30))
                    dk2 = ((k_ev - kbar[lab]) ** 2).sum(-1)
                    s2 = z1.clone().index_add_(0, lab, dk2) / cl / d
                    PRE[s] = (lab, cnt, cl, kbar, vbar, occ, logn, s2)
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
                    AGF["full"] += W @ o_full; AGF["R"] += W @ oR
                    for si, s in enumerate(SCHEMES):
                        if s == "score-oracle":
                            # **1-D k-means（Lloyd）on s_i = aᵀk_i**，不是等量分位桶
                            # （等量桶不最小化 Σ_C Σ(s_i−s̄_C)²）。唯一依赖 q 的分簇。
                            order = sE.argsort()
                            cen = sE[order][(torch.arange(a.K, device=sE.device)
                                             * sE.numel() // a.K).clamp(
                                                 max=sE.numel() - 1)].clone()
                            for _ in range(8):
                                lab = (sE[:, None] - cen[None]).abs().argmin(-1)
                                cc = torch.zeros(a.K, device=sE.device).index_add_(
                                    0, lab, torch.ones_like(sE))
                                cs_ = torch.zeros(a.K, device=sE.device).index_add_(
                                    0, lab, sE)
                                cen = torch.where(cc > 0, cs_ / cc.clamp_min(1.0), cen)
                            lab = (sE[:, None] - cen[None]).abs().argmin(-1)
                            z1 = torch.zeros(a.K, device=sE.device)
                            cnt = z1.clone().index_add_(0, lab, torch.ones_like(sE))
                            cl = cnt.clamp_min(1.0)
                            kbar = torch.zeros(a.K, d, device=sE.device).index_add_(
                                0, lab, k_ev) / cl[:, None]
                            vbar = torch.zeros(a.K, d, device=sE.device).index_add_(
                                0, lab, v_ev) / cl[:, None]
                            occ = cnt > 0
                            logn = torch.where(occ, cl.log(), torch.full_like(cl, -1e30))
                            dk2 = ((k_ev - kbar[lab]) ** 2).sum(-1)
                            s2 = z1.clone().index_add_(0, lab, dk2) / cl / d
                        else:
                            if s not in PRE:
                                continue
                            lab, cnt, cl, kbar, vbar, occ, logn, s2 = PRE[s]
                            z1 = torch.zeros(a.K, device=sE.device)
                        sbar = aq @ kbar.T
                        delta = sE - sbar[lab]
                        # oracle：对实际 query 的精确簇内投影方差 ⇒ **不可部署**，只当上界
                        var_or = z1.clone().index_add_(0, lab, delta * delta) / cl
                        # 可部署，每簇只多 1 个 scalar：aᵀΣ_j a ≈ (trΣ_j/d)·‖a‖²
                        var_iso = s2 * float(aq @ aq)
                        r0 = sbar + logn
                        r2 = sbar + torch.where(occ, logn + 0.5 * var_or, logn)
                        r2i = sbar + torch.where(occ, logn + 0.5 * var_iso, logn)
                        LE0 = torch.logsumexp(r0, -1)
                        LE2 = torch.logsumexp(r2, -1)
                        LE2i = torch.logsumexp(r2i, -1)
                        vw = float((cnt * var_or).sum() / cnt.sum().clamp_min(1))
                        vt = float(sE.var(unbiased=False))
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
                                     float((W @ (cell(LE2i, vh) - o_full)).norm() / nd),
                                     float(LE2i - LE), vw / max(vt, 1e-30), d_eff,
                                     1.0 / (1.0 + e_ * e_)))
                        for key, LEx, vx in (("cc", LE0, vh), ("mc", LE, vh),
                                             ("cd", LE0, vE), ("2or", LE2, vh),
                                             ("2iso", LE2i, vh)):
                            AGG[s][key] += W @ cell(LEx, vx)
                        errs = [si]
                        for gm in GAMMAS:
                            LEg = (LE + np.log(gm)) if gm > 0 else torch.tensor(
                                -1e30, device=LE.device, dtype=LE.dtype)
                            errs.append(float((W @ (cell(LEg, vh) - o_full)).norm() / nd))
                        gam.append(errs)
            dl = AGF["full"] - AGF["R"]
            nl = dl.norm().clamp_min(1e-12)
            for si, s in enumerate(SCHEMES):
                if AGG[s]["cc"].abs().sum() == 0:
                    continue
                lay_rows.append((si, l,
                                 float((AGG[s]["cc"] - AGF["full"]).norm() / nl),
                                 float((AGG[s]["mc"] - AGF["full"]).norm() / nl),
                                 float((AGG[s]["cd"] - AGF["full"]).norm() / nl),
                                 float((AGG[s]["2or"] - AGF["full"]).norm() / nl),
                                 float((AGG[s]["2iso"] - AGF["full"]).norm() / nl),
                                 cs(AGG[s]["cc"] - AGF["R"], dl)))
        for l in ([] if CQ_FROZEN else a.layers):   # 冻结后不再累计
            Aq = _Q[l][0].float() * (d ** -0.5)   # [HQ,T,d]
            G = Aq.shape[0] // H
            for h in range(H):
                z = Aq.view(H, G, -1, d)[h].reshape(-1, d)
                CQ[(l, h)] = CQ.get((l, h), torch.zeros(d, d, device=z.device)) + z.T @ z
                CQN[(l, h)] = CQN.get((l, h), 0) + z.shape[0]
        del kv
        torch.cuda.empty_cache()
        print(f"  样本 {i} 完成，累计 {len(rows)} 行", flush=True)
        # 中途报告 + 落盘：4 小时的任务必须能中途看，且被杀不白跑
        done = i - a.start + 1
        if done % a.report_every == 0 and done < a.n:
            np.save(f"{a.out}_rows.npy", np.array(rows))
            np.save(f"{a.out}_gam.npy", np.array(gam))
            np.save(f"{a.out}_lay.npy",
                    np.array(lay_rows) if lay_rows else np.zeros((0, 8)))
            report(np.array(rows), np.array(gam), np.array(lay_rows) if lay_rows else np.zeros((0,8)), a, deff_cq, done)

    A = np.array(rows); Gm = np.array(gam)
    np.save(f"{a.out}_rows.npy", A); np.save(f"{a.out}_gam.npy", Gm)
    LAY = np.array(lay_rows) if lay_rows else np.zeros((0, 8))
    np.save(f"{a.out}_lay.npy", LAY)
    report(A, Gm, LAY, a, deff_cq, a.n)


if __name__ == "__main__":
    main()
