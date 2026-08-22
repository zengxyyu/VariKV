#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**表达力探针**：Student 的函数族能不能表达教师的目标？——训练之前的闸。

────────────────────────────────────────────────────────────────────────────
问题（外部复核 2026-08-22 指出，读代码逐字确认，采纳）
────────────────────────────────────────────────────────────────────────────
教师（`prometa/teacher.py:future_utility`）：

    U*_i = max_{t, g∈group(h)}  softmax_i( q[g,t]·k_i/√d )        再 `to_dist` 归一化

Student（`prometa/model.py:demand`）：

    Û_i = softmax_i( q̂·k_i/√d )

写成对数形式，包含关系就是可证的：

    log U*_i = max_r ( q_r·k_i − log Z_r ) − log S     ⇒ **k_i 的凸分段线性函数**
    log Û_i  =        q̂ ·k_i − log Z                   ⇒ **k_i 的仿射函数**

⇒ **Student 族 = 教师族的 R=1 特例**，max 真起作用时是真子集。

**已有的免费证书**：令 `S = Σ_i max_r p_{r,i}`。若只有 `R_act` 个 atom 曾取到
max，则 `S ≤ Σ_{r∈act} Σ_i p_{r,i} = R_act` ⇒ **`S` 是活跃 atom 数的下界**；
且 `S = 1` ⟺ 全部 `T·G` 行完全相同。`prometa/train.py:to_dist` 的注释记录
真机行和 **1.8–37.9** ⇒ 目标确实不在 Student 族里。

────────────────────────────────────────────────────────────────────────────
但**分布不可表达 ≠ 决策不可达**
────────────────────────────────────────────────────────────────────────────
驱逐只用 `R_i = ρ_β(U_{·,i})` 的 **top-B 排序**。所以本探针量的是决策级差距：

  ① `S`、逐位 argmax 的**不同 atom 数**、贪心覆盖到 90%/99% 需要几个 atom
     —— 目标有多"多峰"（纯统计，无优化）
  ② **贪心真 atom** 子集 R∈{1,2,4,8,16}：`Û_R = normalize(max_{r∈A} p_r)`
     的 KL 与 top-k Jaccard。目标函数 `f(A)=Σ_i max_{r∈A} p_{r,i}` 单调次模
     ⇒ 贪心有 (1−1/e) 保证。这是**可达下界**（用的是真 query 向量）
  ③ **自由 q oracle（R=1）**：直接对 `q*∈R^d` 优化 `KL(Ū*‖softmax(q*K))`。
     这是**当前架构的天花板**，与训练好坏无关
  ④ 决策级：`ρ_β` 聚合后做全局 top-B，与教师的保留集比 Jaccard

判据（写死，先于结果）：
  ③ 的 `J_decision@0.1` ≳ 0.9  ⇒ 单 q 足够，架构不是瓶颈，照原样训练
  ③ 明显低于 ② 的 R=4          ⇒ **架构就是瓶颈**，必须改成多 atom：
                                  `Û_m = normalize(max_r softmax(q̂_{m,r}K))`
                                  —— 与教师同族，只是把 T·G 换成 R
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "external/FastKVzip/prefill"))


def greedy_atoms(P, R):
    """P: [R0,n] 非负。→ 贪心选出的 R 个 atom 下标（`f(A)=Σ_i max_{r∈A}P` 单调次模）。"""
    R0 = P.shape[0]
    cur = torch.zeros(P.shape[1], device=P.device, dtype=P.dtype)
    chosen = []
    for _ in range(min(R, R0)):
        gain = torch.maximum(P, cur[None]).sum(-1) - cur.sum()   # [R0] 逐 atom 边际增益
        gain[torch.tensor(chosen, dtype=torch.long, device=P.device)] = -1.0 if chosen else gain[0] * 0 - 1.0
        j = int(gain.argmax())
        chosen.append(j)
        cur = torch.maximum(cur, P[j])
    return chosen


def topk_jaccard(a, b, k):
    """a,b: [...,n] → 逐行 top-k 集合的 Jaccard 均值。"""
    n = a.shape[-1]
    k = max(1, min(k, n))
    ia = a.topk(k, -1).indices
    ib = b.topk(k, -1).indices
    A = torch.zeros_like(a, dtype=torch.bool).scatter_(-1, ia, True)
    B = torch.zeros_like(b, dtype=torch.bool).scatter_(-1, ib, True)
    inter = (A & B).sum(-1).float()
    return float((inter / (2 * k - inter).clamp_min(1)).mean())


def _kl(p, q, eps=1e-12):
    return (p * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum(-1)


def free_q(target, K, R=1, steps=400, lr=0.05, seed=0, report=None, mode="free"):
    """target: [B,n]；K: [B,n,d]。→ `Û = normalize(max_r softmax(q_r·K/√d))` 的
    **族内最优点**（自由优化 R 个 atom，与训练无关）。R=1 即当前 Student 架构。

    这与教师**同族**：教师是 `normalize(max_{t,g} softmax(q_{t,g}·K))`，
    只是它的 atom 数 `R0 = T·G` 由数据给定，这里的 `R` 是可选容量。

    ⚠ atom 必须**非对称初始化**（本仓库在 `probe_bias` 上吃过对称鞍点的亏）。
    ⚠ 硬 `max` 的梯度只流向 argmax atom（与 maxpool 同理），这是有意的 ——
      要匹配教师的函数形式，不能换成 soft-max，否则量的是另一个族。
    """
    B, n, d = K.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    sc = 1.0 / (d ** 0.5)
    traj = []
    if mode == "free":
        q = (torch.randn(B, R, d, generator=g) * 0.5).to(K.device).requires_grad_(True)
        prm, mk = [q], (lambda: q)
    elif mode == "shared_off":
        # **真实架构会用的廉价参数化**：`q̂_{l,h,r} = A(u_m) + head_emb[l,h] + A(bias_r)`
        # ⇒ atom 之间只差一个**跨 (层,头) 共享**的常向量。只多 `R×d` 个参数，
        # 而不是把 trunk 的输出宽度乘 R（那要多约 65 万参数）。
        # 若它能吃下自由 q 的多 atom 增益，架构改动就是廉价的。
        base = (torch.randn(B, d, generator=g) * 0.5).to(K.device).requires_grad_(True)
        off = (torch.randn(R, d, generator=g) * 0.5).to(K.device).requires_grad_(True)
        prm, mk = [base, off], (lambda: base[:, None, :] + off[None])
    else:
        raise ValueError(mode)
    opt = torch.optim.Adam(prm, lr=lr)

    def fwd(qq):
        lg = torch.einsum("brd,bnd->brn", qq, K) * sc
        p = torch.softmax(lg, -1)                      # [B,R,n]
        u = p.amax(1)                                  # [B,n] 逐元素 max（同教师）
        return u / u.sum(-1, keepdim=True).clamp_min(1e-30)

    for i in range(steps):
        pred = fwd(mk())
        loss = _kl(target, pred).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if report is not None and (i + 1) % max(1, steps // 4) == 0:
            traj.append(float(loss))
    with torch.no_grad():
        pred = fwd(mk())
    if report is not None:
        report.extend(traj)
    return pred.detach(), _kl(target, pred).detach()


# ═══════════════════════════════════════════════════════════════════════════
def _selftest():
    """阴性/阳性对照：**探针本身**必须能区分「单 atom」与「双 atom」。"""
    torch.manual_seed(0)
    d, n = 16, 400
    K = torch.randn(n, d)

    # ① 单 atom：S 必须恰为 1，R=1 必须完美
    q1 = torch.randn(d)
    P1 = torch.softmax(q1 @ K.T / d ** 0.5, -1)[None]        # [1,n]
    U1 = P1.max(0).values
    assert abs(float(U1.sum()) - 1.0) < 1e-5, float(U1.sum())
    pred, kl = free_q(U1[None] / U1.sum(), K[None], R=1, steps=500, lr=0.1)
    j1 = topk_jaccard(U1[None], pred, k=int(0.1 * n))
    assert kl.item() < 1e-3 and j1 > 0.99, (kl.item(), j1)
    print(f"① 单 atom：S={float(U1.sum()):.6f}（=1）、自由 q 的 KL={kl.item():.2e}、"
          f"J@0.1={j1:.4f}　PASS")

    # ② 双 atom **正交且尖锐** ⇒ S 明显 >1，且 R=1 必须做不到
    #    （构造成两个互相远离的峰；单个 softmax 无法同时给两处高质量）
    a, b = torch.randn(d), torch.randn(d)
    b = b - (b @ a) / (a @ a) * a
    sharp = 6.0
    P2 = torch.stack([torch.softmax(sharp * a @ K.T / d ** 0.5, -1),
                      torch.softmax(sharp * b @ K.T / d ** 0.5, -1)])
    U2 = P2.max(0).values
    S2 = float(U2.sum())
    T2 = (U2 / U2.sum())[None]
    pred1, kl1 = free_q(T2, K[None], R=1, steps=800, lr=0.1)
    pred2, kl2f = free_q(T2, K[None], R=2, steps=800, lr=0.1)
    g = greedy_atoms(P2, 2)
    U2g = P2[g].max(0).values; U2g = U2g / U2g.sum()
    kl2 = float(_kl(T2[0], U2g))
    j_1 = topk_jaccard(T2, pred1, k=int(0.1 * n))
    j_2 = topk_jaccard(T2, U2g[None], k=int(0.1 * n))
    assert S2 > 1.2, S2
    assert kl1.item() > 10 * max(kl2, 1e-9), (kl1.item(), kl2)
    j_2f = topk_jaccard(T2, pred2, k=int(0.1 * n))
    assert kl2f.item() < 0.5 * kl1.item(), (kl1.item(), kl2f.item())
    print(f"② 双 atom：S={S2:.3f}（>1 即证书成立）；自由 q(R=1) KL={kl1.item():.4f} "
          f"J@0.1={j_1:.4f}　→　**自由 q(R=2) KL={kl2f.item():.4f} J@0.1={j_2f:.4f}**"
          f"　（贪心真 atom R=2 KL={kl2:.2e} J@0.1={j_2:.4f}）　PASS")

    # ③ 贪心确实挑出那两个 atom（不是随便两个）
    assert sorted(g) == [0, 1], g
    print(f"③ 贪心在 R0=2 上选出 {g}　PASS")

    # ④ topk_jaccard 的边界：自己对自己 = 1，互补 = 0
    x = torch.rand(3, 100)
    assert abs(topk_jaccard(x, x, 10) - 1.0) < 1e-6
    assert topk_jaccard(x, -x, 10) < 0.05
    print("④ topk_jaccard 自反=1、反序≈0　PASS")
    print("\nscratch_prometa_repr.py 自测 4 条全过")


# ═══════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("-g", "--gate", default="fastkvzip")
    p.add_argument("--manifest", default="prometa_data/manifest_v1_ss.jsonl")
    p.add_argument("--split", default="val")
    p.add_argument("--n_docs", type=int, default=3)
    p.add_argument("--chunk", type=int, default=16000)
    p.add_argument("--window", type=int, default=4096)
    p.add_argument("--span", default="q")
    p.add_argument("--Rs", default="1,2,4,8,16")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--rhos", default="0.02,0.05,0.1,0.2")
    p.add_argument("--skip_greedy", action="store_true",
                   help="跳过贪心真 atom 那一支（它的内层循环是 GPU 同步大户）。"
                        "上一轮已测得贪心是弱基线：R=16 才追平自由 q 的 R=1")
    p.add_argument("--out", default="scratch_prometa_repr.json")
    a = p.parse_args()
    if a.selftest:
        return _selftest()

    from model.wrapper import ModelKVzip
    from prometa.teacher import chunk_ranges, future_utility

    Rs = [int(x) for x in a.Rs.split(",")]
    rhos = [float(x) for x in a.rhos.split(",")]
    recs = [json.loads(l) for l in open(a.manifest)]
    recs = [r for r in recs if r["split"] == a.split and r["futures"]][:a.n_docs]
    print(f"[repr] {len(recs)} 篇（split={a.split}）：{[r['n_ctx'] for r in recs]}", flush=True)

    model = ModelKVzip(a.model, "retain", a.gate)
    dev = model.device
    sys_len = int(model.sys_prompt_ids.shape[1])
    rows, cross = [], {}

    for ri, rec in enumerate(recs):
        ctx = torch.tensor([rec["ctx"]], device=dev)
        n_tot = ctx.shape[1] + sys_len
        _all, usable = chunk_ranges(n_tot, sys_len, a.chunk, a.window)
        if not usable:
            print(f"  doc{ri}: 无可用 chunk，跳过"); continue
        lo, hi = usable[len(usable) // 2]           # 取中间那块，避开首尾特例
        teas, fits = [], []
        prev = getattr(model, "varikv_train", False); model.varikv_train = True
        try:
            with torch.no_grad():
                kv = model.prefill(ctx, prefill_chunk_size=a.chunk,
                                   do_score=False, chunk_ratio=1.0)
        finally:
            model.varikv_train = prev
        L, Hkv = len(kv.key_cache), int(kv.key_cache[0].shape[1])
        n = hi - lo
        d = int(kv.key_cache[0].shape[-1])
        Kc = torch.stack([kv.key_cache[l][0, :, lo:hi, :].float() for l in range(L)])  # [L,H,n,d]

        for mi, f in enumerate(rec["futures"]):
            ids = list(f["q"]) + (list(f["a"]) if a.span == "qa" else [])
            with torch.no_grad():
                kv.capture_q, kv._q_cap = True, {}
                model(torch.tensor([ids], device=dev), kv, update_cache=False)
                kv.capture_q = False
                assert len(kv._q_cap) == L
                qc = [kv._q_cap[l] for l in range(L)]
                Utea = future_utility(kv.key_cache, qc, lo, hi, out_np=False)   # [L,H,n] 未归一
            G = qc[0].shape[1] // Hkv
            T = qc[0].shape[2]
            S = Utea.sum(-1)                                     # [L,H] 证书
            Tgt = (Utea / S[..., None].clamp_min(1e-30))         # 归一化教师

            # ── ①② 逐 (l,h) 的多峰统计 + 贪心真 atom ──────────────────────
            n_dist, n90, n90_cens = [], [], [0]
            Ug = {R: torch.zeros(L, Hkv, n, device=dev) for R in Rs}
            for l in range(L):
                Kl = Kc[l]                                        # [H,n,d]
                ql = qc[l][0].float().reshape(Hkv, G, T, d)       # [H,G,T,d]
                for h in range(Hkv):
                    P = torch.softmax(torch.einsum("rd,nd->rn",
                                                   ql[h].reshape(G * T, d), Kl[h]) / d ** 0.5, -1)
                    am = P.argmax(0)
                    n_dist.append(int(torch.unique(am).numel()))
                    # 贪心固定跑 `CAP` 轮、把覆盖曲线**留在 GPU 上**，最后一次性
                    # 同步。原写法每轮 `float(cur.sum())` ⇒ 最坏 29 万次 GPU 同步
                    # （本仓库「不要在枚举里 .item()」那条规矩，第二次踩）。
                    if a.skip_greedy:
                        n90.append(0)
                        for R in Rs:
                            Ug[R][l, h] = P.max(0).values / P.max(0).values.sum().clamp_min(1e-30)
                        del P; continue
                    CAP = min(G * T, max(Rs) if max(Rs) >= 32 else 32)
                    cur = torch.zeros(n, device=dev)
                    cov = torch.empty(CAP, device=dev)
                    chosen, mask = [], torch.zeros(G * T, dtype=torch.bool, device=dev)
                    for c in range(CAP):
                        gain = torch.maximum(P, cur[None]).sum(-1)
                        gain = gain.masked_fill(mask, float("-inf"))
                        j = int(gain.argmax())          # 每轮仅此一次同步，不可避免
                        chosen.append(j); mask[j] = True
                        cur = torch.maximum(cur, P[j]); cov[c] = cur.sum()
                    covc = cov.cpu().numpy(); tot = float(P.max(0).values.sum())
                    hit = np.nonzero(covc >= 0.90 * tot)[0]
                    # **删失要标出来**：没在 CAP 轮内到 90% 的记成 CAP 并计数，
                    # 不能当成「只需要 CAP 个」——那会把多峰程度系统性低估。
                    if len(hit):
                        n90.append(int(hit[0]) + 1)
                    else:
                        n90.append(CAP); n90_cens[0] += 1
                    for R in Rs:
                        g = chosen[:R]
                        u = P[g].max(0).values; u = u / u.sum().clamp_min(1e-30)
                        Ug[R][l, h] = u
                    del P
            # ── ③ 自由 q oracle（R=1，当前架构的天花板），批到 (l,h) 上 ──────
            tgt = Tgt.reshape(L * Hkv, n)
            Kf = Kc.reshape(L * Hkv, n, d)
            preds, kls, convs = {}, {}, {}
            spreds, skls = {}, {}
            for R in Rs:
                tr = []
                pr, kk = free_q(tgt, Kf, R=R, steps=a.steps, report=tr)
                preds[R] = pr.reshape(L, Hkv, n); kls[R] = float(kk.mean()); convs[R] = tr
                if R > 1:
                    sp, sk = free_q(tgt, Kf, R=R, steps=a.steps, mode="shared_off")
                    spreds[R] = sp.reshape(L, Hkv, n); skls[R] = float(sk.mean())
            pred1 = preds[Rs[0]]

            row = dict(doc=ri, m=mi, kind=f["kind"], n=n, T=T, G=G, R0=G * T,
                       S_mean=float(S.mean()), S_med=float(S.median()), S_max=float(S.max()),
                       n_argmax_distinct_med=float(np.median(n_dist)),
                       n_atoms_90_med=float(np.median(n90)), n_atoms_90_max=int(max(n90)),
                       n90_censored=int(n90_cens[0]), n90_cap=int(min(G * T, max(32, max(Rs)))),
                       KL_freeq_R1=kls[Rs[0]],
                       KL_trivial=float(_kl(tgt, torch.full_like(tgt, 1.0 / n)).mean()))
            for R in Rs:
                row[f"KL_greedy_R{R}"] = float(_kl(Tgt, Ug[R]).mean())
                row[f"KL_freeq_R{R}"] = kls[R]
                if R in skls: row[f"KL_shared_R{R}"] = skls[R]
                row[f"conv_R{R}"] = convs[R]        # 收敛轨迹（四个检查点）
            for r in rhos:
                k = max(1, int(round(r * n)))
                for R in Rs:
                    row[f"J_freeq_R{R}@{r}"] = topk_jaccard(Tgt.reshape(-1, n),
                                                            preds[R].reshape(-1, n), k)
                    row[f"J_greedy_R{R}@{r}"] = topk_jaccard(Tgt.reshape(-1, n),
                                                             Ug[R].reshape(-1, n), k)
                    if R in spreds:
                        row[f"J_shared_R{R}@{r}"] = topk_jaccard(
                            Tgt.reshape(-1, n), spreds[R].reshape(-1, n), k)
                row[f"J_freeq_R1@{r}"] = row[f"J_freeq_R{Rs[0]}@{r}"]
            teas.append(Tgt.reshape(-1, n).cpu())    # 跨未来参照要用
            fits.append(preds[Rs[0]].reshape(-1, n).cpu())
            rows.append(row)
            print(f"  doc{ri} m{mi} {f['kind']:<12} T={T} R0={G*T:<4} "
                  f"S(中位)={row['S_med']:.2f} 不同argmax(中位)={row['n_argmax_distinct_med']:.0f} "
                  f"90%需atom(中位)={row['n_atoms_90_med']:.0f} | "
                  f"KL 平凡{row['KL_trivial']:.3f} " +
                  " ".join(f"自由q R{R} {kls[R]:.3f}" for R in Rs) + " | J@0.1 " +
                  " ".join(f"R{R} {row[f'J_freeq_R{R}@0.1']:.3f}" for R in Rs),
                  flush=True)
            del Utea, Tgt, Ug, pred1, qc
            torch.cuda.empty_cache()
        # ── 跨未来参照：不同未来的**教师目标之间**本来就有多像？ ─────────────
        # 若它 ≈ 单 q 拟合的 J，说明单 q 抓到的**全是跨未来共享的结构**，
        # 恰好丢掉 ProMeta 唯一需要的那部分（逐未来差异）。这个参照不做，
        # 「J=0.50 算高算低」就无从判断（不许报没有参照的数）。
        if len(teas) >= 2:
            for r_ in rhos:
                k_ = max(1, int(round(r_ * n)))
                cx_t = [topk_jaccard(teas[i], teas[j], k_)
                        for i in range(len(teas)) for j in range(i + 1, len(teas))]
                cx_f = [topk_jaccard(fits[i], fits[j], k_)
                        for i in range(len(fits)) for j in range(i + 1, len(fits))]
                cross.setdefault(r_, []).append((float(np.mean(cx_t)), float(np.mean(cx_f))))
        del kv, Kc; torch.cuda.empty_cache()

    json.dump(rows, open(a.out, "w"), indent=1)
    if not rows:
        print("没有任何结果"); return
    def agg(k): return float(np.mean([r[k] for r in rows if k in r]))
    print("\n" + "=" * 74)
    print(f"汇总（{len(rows)} 个 (doc,future)，每个含 {rows[0]['n']} 位置 × 112 个 (层,头)）")
    print(f"  证书 S 中位 {agg('S_med'):.2f}（=1 才表示单 atom；>1 即证明目标不在 R=1 族里）")
    print(f"  逐位 argmax 的不同 atom 数（中位）{agg('n_argmax_distinct_med'):.1f} / R0={rows[0]['R0']}")
    _cens = sum(r["n90_censored"] for r in rows); _tot = len(rows) * 112
    if a.skip_greedy:
        print("  贪心覆盖统计：**本轮 --skip_greedy 未测**（上一轮实测 >32 个 atom，99.6% 删失）")
    else:
        print(f"  贪心覆盖 90% 质量需要的 atom 数（中位）{agg('n_atoms_90_med'):.1f}"
              f"（上限 {rows[0]['n90_cap']} 轮内没到 90% 的有 {_cens}/{_tot} = {_cens/max(_tot,1):.1%}"
              f" ⇒ 这部分是**删失值，真值更大**）")
    print(f"\n  KL：平凡解 {agg('KL_trivial'):.4f} | **自由 q(R=1) 天花板 {agg('KL_freeq_R1'):.4f}**")
    for R in Rs:
        extra = f"　共享偏移 {agg(f'KL_shared_R{R}'):.4f}" if R > 1 else ""
        print(f"        自由 q R={R:<3} {agg(f'KL_freeq_R{R}'):.4f}{extra}")
    for r in rhos:
        rnd = r / (2 - r)          # 随机选 k 个的期望 Jaccard = k/(2n-k)
        line = f"  J@{r} (随机基线 {rnd:.4f})：自由q"
        for R in Rs:
            line += f" R{R} {agg(f'J_freeq_R{R}@{r}'):.4f}"
        line += " ｜ 共享偏移"
        for R in Rs[1:]:
            line += f" R{R} {agg(f'J_shared_R{R}@{r}'):.4f}"
        if r in cross:
            ct = float(np.mean([x[0] for x in cross[r]]))
            cf = float(np.mean([x[1] for x in cross[r]]))
            line += f" ｜ **跨未来 教师-教师 {ct:.4f} / 拟合-拟合 {cf:.4f}**"
        print(line)

    # ── 判词由数字生成；参照必须是**同族的 R>1**，不是弱基线贪心 ─────────────
    j1 = agg(f"J_freeq_R{Rs[0]}@0.1"); jmax = agg(f"J_freeq_R{Rs[-1]}@0.1")
    ct = float(np.mean([x[0] for x in cross[0.1]])) if 0.1 in cross else float("nan")
    print(f"\n判据（全部在 ρ=0.1 上）")
    print(f"  自由 q R={Rs[0]} = {j1:.4f}  →  R={Rs[-1]} = {jmax:.4f}"
          f"（增益 {jmax-j1:+.4f}）；随机 {0.1/1.9:.4f}；跨未来教师-教师 {ct:.4f}")
    if jmax - j1 >= 0.05:
        print(f"⇒ **加 atom 有实质增益（{jmax-j1:+.4f}）⇒ 当前 R=1 架构是瓶颈**，"
              f"应改成 Û=normalize(max_r softmax(q̂_r·K))，与教师同族")
    else:
        print(f"⇒ 加 atom 到 R={Rs[-1]} 只多 {jmax-j1:+.4f} ⇒ **瓶颈不在 atom 数**；"
              f"若 J 绝对值仍低，要查的是别处（K 几何 / 目标本身的熵）")
    if not np.isnan(ct):
        print(f"⇒ 单 q 的 J({j1:.4f}) vs 跨未来教师互相的 J({ct:.4f})："
              + ("**单 q 没有超过『不同未来本来就长得像』的水平 ⇒ 它抓到的基本是共享结构，"
                 "而 ProMeta 要的恰是逐未来差异**" if j1 <= ct + 0.02 else
                 f"单 q 超出共享水平 {j1-ct:+.4f} ⇒ 确实抓到了一些逐未来的信息"))
    print("Finished.")


if __name__ == "__main__":
    main()
