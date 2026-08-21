"""任务基准的**反事实配额优势教师**（2026-08-21）。

────────────────────────────────────────────────────────────────────────────
它解决什么
────────────────────────────────────────────────────────────────────────────
现有两种教师都以**满缓存注意力输出**为参照：

    full_single :  U_i = err(S \\ i) − err(S)            条件=满缓存，参照=满缓存
    set_marginal:  U_i = err(S \\ i) − err(S ∪ {i})      条件=真实存活集合，参照=满缓存

但实测有 **28/77 格 headroom < 0**（压缩比满缓存还好），在那些格上
「向满缓存靠拢」**方向就是错的**。而 `chr03`（在评测工作点重训）仍 −12.87、
损伤占 Δ 方差 **86%** —— 都指向参照系而非条件。

本教师换掉参照系：

    J(b)          = − NLL(答案 token | 上下文, 配额 b)          ← **任务效用**
    A_{i←j}^{(k)} = J(b⁰ + k·e_i − k·e_j) − J(b⁰)              ← **相对当前基线**

预算严格守恒：受主 i 加 k 个、施主 j 减 k 个，`Σ_h b_h` 不变（脚本内断言）。
答案由 `make_retrieval` **构造时已知**，所以标签不需要人工标注，且是稠密的
对数概率而非二值命中。

────────────────────────────────────────────────────────────────────────────
为什么便宜：改配额**不需要重新预填**
────────────────────────────────────────────────────────────────────────────
`RetainCache` 物理上保留全部 KV，保留集只是 `self.valid` 这个掩码
（`prepare()` 用 `_get_valid()` 挑进注意力）。所以：

    预填一次（贵）  →  换掩码 → 前向答案 token（便宜）→ `slice()` 回滚  → 重复

每个动作只多一次 ~30 token 的前向。

────────────────────────────────────────────────────────────────────────────
三个自检（都会打印，任一失败即中止）
────────────────────────────────────────────────────────────────────────────
① **零动作**（k=0）必须给出 `A == 0`（逐位），否则说明掩码写入/回滚有副作用；
② **预算守恒**：每个动作后 `Σ_h b_h` 必须与基线相同；
③ **信度**：把答案 token 前后对半分别算 A，报两半的相关与符号一致率 ——
   若两半都对不上，这个标签就不能拿来训练（本项目 `U^NLL` 那次正是栽在没测这个）。
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))

from model import ModelKVzip                                        # noqa: E402
from data.load import load_fineweb                                  # noqa: E402


# ────────────────────────────── 任务构造 ──────────────────────────────
def make_task(m, ids, max_ctx, window, n_dup, rng):
    """插入合成事实，返回 (上下文 ids, 问句 ids, 答案 ids, meta)。

    与 `scratch_ctrl_teacher.make_retrieval` **同构**（同样的 key/val 格式、
    同样的插入区间与倒序插入），差别只有一处：这里把**问句**与**答案**分开返回，
    因为 J 只在答案 token 上算 NLL —— 问句是条件不是预测目标。
    """
    ctx = list(ids[-max_ctx:])
    hx = "0123456789abcdef"
    key = "".join(rng.choices(hx, k=16))
    val = "".join(rng.choices(hx, k=16))
    fact = m.encode(f" The secret key {key} maps to the value {val}. ")[0].tolist()
    q_ids = m.encode(f"\nQuestion: What value does the secret key {key} "
                     f"map to?\nAnswer:")[0].tolist()
    a_ids = m.encode(f" {val}")[0].tolist()
    lo, hi = int(0.05 * (len(ctx) - window)), int(0.90 * (len(ctx) - window))
    pos = sorted(rng.sample(range(lo, hi), n_dup), reverse=True)
    for p_ in pos:
        ctx[p_:p_] = fact
    return ctx, q_ids, a_ids, dict(key=key, val=val, pos=sorted(pos), n_dup=n_dup)


# ────────────────────────────── 效用 J ──────────────────────────────
@torch.no_grad()
def answer_nll(m, kv, q_t, a_t, n_seen, halves=False):
    """→ 答案 token 的平均 NLL（越小越好）。`halves=True` 时另返回前/后半。

    **必须回滚**：`m.model(...)` 会把 query/answer 的 K/V 追加进 cache，
    不回滚则下一个动作看到的上下文已被污染（`teacher_state` 的注释记过这个坑）。
    """
    inp = torch.cat([q_t, a_t], dim=1)
    out = m.model(inp, past_key_values=kv, use_cache=True)
    kv.slice(n_seen)                                   # 回滚到纯上下文
    n_a = a_t.shape[1]
    # 预测第 t 个 token 的 logits 在位置 t−1；答案占 inp 的最后 n_a 个位置
    lg = out.logits[0, -n_a - 1:-1].float()
    tg = a_t[0]
    nll = F.cross_entropy(lg, tg, reduction="none")    # [n_a]
    if not halves:
        return float(nll.mean()), None, None
    h = n_a // 2
    return float(nll.mean()), float(nll[:h].mean()), float(nll[h:].mean())


# ────────────────────────────── 动作构造 ──────────────────────────────
def apply_transfer(valid, score, i, j, k):
    """在 `valid` 的副本上执行「从 j 拿 k 个给 i」。

    受主 i：把它**被驱逐者里分数最高的 k 个**置 True；
    施主 j：把它**保留者里分数最低的 k 个**置 False。
    —— 这是「最小代价的施予/最大收益的接收」，与 `frontier` 探针同一约定。

    返回 (新 valid, 实际转移数)。若任一侧不足 k，按较小者转移以**严格守恒预算**。
    """
    v = valid.clone()
    li, hi_ = i
    lj, hj = j
    ev = (~v[li, hi_]).nonzero(as_tuple=True)[0]
    rt = v[lj, hj].nonzero(as_tuple=True)[0]
    kk = int(min(k, len(ev), len(rt)))
    if kk == 0:
        return v, 0
    add = ev[torch.argsort(score[li, hi_][ev], descending=True)[:kk]]
    rem = rt[torch.argsort(score[lj, hj][rt])[:kk]]
    v[li, hi_, add] = True
    v[lj, hj, rem] = False
    return v, kk


def pick_actions(valid, score, n_recv, n_don, rng):
    """挑候选受主/施主。

    受主：**最好的被驱逐者分数**最高的头 —— 它最"憋屈"；
    施主：**最差的保留者分数**最低的头 —— 它最"浪费"。
    这是纯启发式，只用来把 G² 个 pair 砍到 n_recv×n_don；
    另加 `n_recv` 个**随机受主**作对照，防止启发式本身把结论选出来。
    """
    L, H, _ = valid.shape
    best_ev, worst_rt = [], []
    for l in range(L):
        for h in range(H):
            ev = (~valid[l, h]).nonzero(as_tuple=True)[0]
            rt = valid[l, h].nonzero(as_tuple=True)[0]
            best_ev.append(float(score[l, h][ev].max()) if len(ev) else -1e9)
            worst_rt.append(float(score[l, h][rt].min()) if len(rt) else 1e9)
    best_ev = np.array(best_ev); worst_rt = np.array(worst_rt)
    recv = list(np.argsort(-best_ev)[:n_recv])
    don = list(np.argsort(worst_rt)[:n_don])
    pool = [g for g in range(L * H) if worst_rt[g] < 1e8]
    rnd_recv = rng.sample(pool, min(n_recv, len(pool)))
    return ([(int(g) // H, int(g) % H) for g in recv],
            [(int(g) // H, int(g) % H) for g in don],
            [(int(g) // H, int(g) % H) for g in rnd_recv])


# ────────────────────────────── 主流程 ──────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--gate", default="fastkvzip")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--chunk", type=int, default=16000)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--level", default="pair")
    ap.add_argument("--max_ctx", type=int, default=131072)
    ap.add_argument("--n_doc", type=int, default=2)
    ap.add_argument("--n_dup", type=int, default=1)
    ap.add_argument("--n_recv", type=int, default=4)
    ap.add_argument("--n_don", type=int, default=4)
    ap.add_argument("--ks", default="1,4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="scratch_adv_probe.json")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]

    m = ModelKVzip(a.model, "retain", a.gate)
    L = m.config.num_hidden_layers
    H = m.config.num_key_value_heads
    texts = [d["context"] for d in load_fineweb("fineweb_10k_cat")][:a.n_doc]
    print(f"文档 {len(texts)} 篇  L={L} H={H}  ρ={a.ratio}  k∈{ks}", flush=True)

    recs = []
    for di, txt in enumerate(texts):
        ids = m.encode(txt)[0].tolist()
        rng = random.Random(a.seed * 1000 + di)
        ctx_ids, q_ids, ans_ids, meta = make_task(
            m, ids, a.max_ctx, a.window, a.n_dup, rng)
        if a.ratio * len(ctx_ids) <= a.window:
            print(f"doc{di}: clen={len(ctx_ids)} 会塌缩到 chunk_ratio=0，跳过")
            continue
        ctx_t = torch.tensor([ctx_ids], device=m.device)
        q_t = torch.tensor([q_ids], device=m.device)
        a_t = torch.tensor([ans_ids], device=m.device)

        kv = m.prefill(ctx_t, prefill_chunk_size=a.chunk, do_score=True,
                       chunk_ratio=a.ratio, window_size=a.window, level=a.level)
        n_seen = kv._seen_tokens
        raw_valid = kv.valid                                # 形状见下
        # **两处必须实测而非假设，第二遍复查各抓到一个 bug：**
        # ① `self.valid` 是 `threshold(score)` 的输出，而 score 是 [L,B,H,n]
        #    ⇒ valid 也是 **[L,B,H,n]**（B=1），不是 [L,H,n]。按后者索引会取错头。
        # ② `RetainCache.prune_chunk` 用 `torch.cat` **累积**各 chunk 的 valid，
        #    而末尾的 local window 从不进 evict_range ⇒ **valid 短于 ctx_len**。
        #    `_get_valid` 靠右侧 pad(True) 补齐。所以 score 必须按 valid 的实际
        #    长度切，不能按 [start_idx:end_idx]。
        vshape = tuple(raw_valid.shape)
        assert raw_valid.dim() in (3, 4), vshape
        base_valid = (raw_valid[:, 0] if raw_valid.dim() == 4 else raw_valid).clone()
        n_ev = base_valid.shape[-1]                          # 实际被驱逐区长度
        sc = torch.stack(kv.score, 0)
        sc = (sc[:, 0] if sc.dim() == 4 else sc).float()
        sc = sc[..., kv.start_idx:kv.start_idx + n_ev].contiguous()
        assert sc.shape == base_valid.shape, (sc.shape, base_valid.shape, vshape)
        B0 = int(base_valid.sum())

        j0, j0a, j0b = answer_nll(m, kv, q_t, a_t, n_seen, halves=True)
        print(f"doc{di}: clen={len(ctx_ids)} 保留 {B0} "
              f"({B0/base_valid.numel():.3f})  基线 NLL {j0:.4f}  "
              f"答案 {len(ans_ids)} tok", flush=True)

        # ---- 自检 ①：零动作必须逐位复现基线 ----
        kv.valid = (base_valid[:, None] if raw_valid.dim() == 4
                    else base_valid).clone()
        j_null, _, _ = answer_nll(m, kv, q_t, a_t, n_seen)
        assert j_null == j0, f"零动作不复现基线：{j_null} vs {j0}"
        print(f"  自检① 零动作 A = {j0 - j_null:+.3e}（须恰为 0）✓", flush=True)

        recv, don, rnd = pick_actions(base_valid, sc, a.n_recv, a.n_don, rng)
        acts = [(i, j, k, "heur") for i in recv for j in don for k in ks]
        acts += [(i, j, k, "rand") for i in rnd for j in don[:1] for k in ks]

        for (i, j, k, tag) in acts:
            if i == j:
                continue
            v, kk = apply_transfer(base_valid, sc, i, j, k)
            if kk == 0:
                continue
            assert int(v.sum()) == B0, f"预算不守恒 {int(v.sum())} vs {B0}"  # 自检②
            kv.valid = v[:, None] if raw_valid.dim() == 4 else v
            jj, ja, jb = answer_nll(m, kv, q_t, a_t, n_seen, halves=True)
            recs.append(dict(doc=di, recv=list(i), don=list(j), k=kk, tag=tag,
                             A=j0 - jj,            # J = −NLL ⇒ A = NLL0 − NLL'
                             A_h1=j0a - ja, A_h2=j0b - jb))
        kv.valid = raw_valid
        del kv
        torch.cuda.empty_cache()

    if not recs:
        print("没有任何动作被评估"); return
    A = np.array([r["A"] for r in recs])
    h1 = np.array([r["A_h1"] for r in recs]); h2 = np.array([r["A_h2"] for r in recs])
    from scipy import stats as st
    print(f"\n=== {len(A)} 个动作 ===")
    print(f"  A 均值 {A.mean():+.5f}  sd {A.std():.5f}  为正 {np.mean(A>0):.1%}")
    print(f"  |A| 中位 {np.median(np.abs(A)):.5f}  最大 {np.abs(A).max():.5f}")
    print(f"\n=== 自检③ 信度（答案 token 前后半各算一次 A）===")
    print(f"  Pearson  {st.pearsonr(h1,h2)[0]:+.3f}   Spearman {st.spearmanr(h1,h2)[0]:+.3f}")
    nz = (h1 != 0) & (h2 != 0)
    print(f"  符号一致率 {np.mean(np.sign(h1[nz])==np.sign(h2[nz])):.1%}"
          f"（n={nz.sum()}）")
    print(f"  ⇒ 若显著低于 ~70%，这个标签**噪声主导，不能拿来训练**")
    hp = [r for r in recs if r["tag"] == "heur"]; rp = [r for r in recs if r["tag"] == "rand"]
    if hp and rp:
        print(f"\n=== 启发式受主 vs 随机受主（对照）===")
        print(f"  启发式 A 均值 {np.mean([r['A'] for r in hp]):+.5f} (n={len(hp)})")
        print(f"  随机   A 均值 {np.mean([r['A'] for r in rp]):+.5f} (n={len(rp)})")
    json.dump(recs, open(os.path.join(ROOT, a.out), "w"))
    print(f"\n写出 {a.out}")


if __name__ == "__main__":
    main()
