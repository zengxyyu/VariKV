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
from data.load import load_fineweb
from attention.kvcache import RetainCache                                  # noqa: E402


# ────────────────────────────── 任务构造 ──────────────────────────────
def make_task(m, ids, max_ctx, window, n_fact, rng):
    """插入 `n_fact` 条**互不相同**的合成事实，返回 (上下文, target, 答案掩码, meta)。

    与 `scratch_ctrl_teacher.make_retrieval` 同构（同样的 key/val 格式、同样的
    插入区间与倒序插入），两处差别：

    ① **多事实多问句**。首跑实测：单条事实的答案只有 12–16 个 token，
       信度检验的符号一致率 **48.7%（抛硬币）**。但那个 48.7% **不证明真效应
       为零** —— 前半/后半答案 token 不是同一潜在量的重复测量（教师强制下后
       半条件于已给出的答案前缀，对缓存的依赖本来就更弱），所以它只说明
       **单问句估计量方差过大**。要估的其实是对未来查询的期望：

           Ā_{i←j} = E_{q∼Q(x)} [ J(q, S') − J(q, S) ]

       因此靠**多个互不相同的问句**求平均，而不是把一个答案拉长。
    ② **返回 `qid`（每个 target 位置属于哪个问句）而不是二值掩码**。所有问答
       拼成一条 target 走**一次**前向 —— 若每条事实各跑一次前向，每个动作的
       成本会乘以 `n_fact`，那是负担不起的。有了 `qid` 就能算**逐问句** NLL，
       信度检验才能按「问句」而不是按「token 位置」对半分。

    **代价（必须记住）**：拼在一条 target 里 ⇒ 第 m 个问句条件于前 m−1 组问答。
    所以 M 个问句不是独立样本，有效 M 小于名义值，且存在顺序效应。信度检验
    因此取**奇数问句 vs 偶数问句**（交错），让两半的平均位置尽量相同。
    """
    ctx = list(ids[-max_ctx:])
    hx = "0123456789abcdef"
    facts, tgt, qid = [], [], []
    for qi in range(n_fact):
        key = "".join(rng.choices(hx, k=16))
        val = "".join(rng.choices(hx, k=16))
        facts.append((key, val,
                      m.encode(f" The secret key {key} maps to the value {val}. ")[0].tolist()))
        q = m.encode(f"\nQuestion: What value does the secret key {key} "
                     f"map to?\nAnswer:")[0].tolist()
        a = m.encode(f" {val}")[0].tolist()
        tgt += q + a
        qid += [-1] * len(q) + [qi] * len(a)
    # 插入点分散在可驱逐区（末尾 window 恒保留，插那里等于没驱逐）；倒序插以免下标失效
    lo, hi = int(0.05 * (len(ctx) - window)), int(0.90 * (len(ctx) - window))
    pos = sorted(rng.sample(range(lo, hi), n_fact), reverse=True)
    for p_, (_, _, f_) in zip(pos, facts):
        ctx[p_:p_] = f_
    assert len(tgt) == len(qid) and sum(x >= 0 for x in qid) > 0
    return ctx, tgt, qid, dict(n_fact=n_fact, pos=sorted(pos),
                               n_ans=int(sum(x >= 0 for x in qid)), n_tgt=len(tgt))


# ────────────────────────────── 效用 J ──────────────────────────────
@torch.no_grad()
def answer_nll(m, kv, t_t, qid_t, n_q, n_seen):
    """→ `(全局平均 NLL, 逐问句平均 NLL 向量[n_q])`，越小越好。

    问句 token 不进损失 —— 它是条件不是预测目标（`qid = −1`）。
    **逐问句**而不是逐 token 求平均：每个问句等权，答案长度的随机波动不会
    让某一条事实主导，且 bootstrap 可以直接对「问句」重采样。

    **必须回滚**：`m.model(...)` 会把 target 的 K/V 追加进 cache，
    不回滚则下一个动作看到的上下文已被污染（`teacher_state` 的注释记过这个坑）。

    对齐：预测 `inp[t]` 用 `logits[t-1]`，所以取 `logits[:-1]` 对 `inp[1:]`。
    """
    out = m.model(t_t, past_key_values=kv, use_cache=True)
    kv.slice(n_seen)                                   # 回滚到纯上下文
    lg = out.logits[0, :-1].float()                    # 预测 inp[1:]
    tg = t_t[0, 1:]
    qq = qid_t[1:]                                     # 与 tg 对齐
    mk = qq >= 0
    nll = F.cross_entropy(lg, tg, reduction="none")[mk]
    qs = qq[mk]
    tot = torch.zeros(n_q, device=nll.device, dtype=nll.dtype).index_add_(0, qs, nll)
    cnt = torch.zeros(n_q, device=nll.device, dtype=nll.dtype).index_add_(
        0, qs, torch.ones_like(nll))
    assert (cnt > 0).all(), f"有问句没有答案 token: {cnt.tolist()}"
    per = (tot / cnt).cpu().numpy()
    return float(per.mean()), per


# ────────────────────────────── 动作构造 ──────────────────────────────
def apply_one_side(valid, score, g, k, add):
    """只动一侧：`add=True` 给 g 加 k 个最好的被驱逐者，否则减 k 个最差的保留者。

    **预算故意不守恒** —— 它只用来做分解项 `J(S∪{i})` / `J(S\\{j})`，
    以便量出交互项 `I_ij`，不作为动作本身。
    """
    v = valid.clone()
    l, h = g
    if add:
        ev = (~v[l, h]).nonzero(as_tuple=True)[0]
        kk = int(min(k, len(ev)))
        if kk:
            v[l, h, ev[torch.argsort(score[l, h][ev], descending=True)[:kk]]] = True
    else:
        rt = v[l, h].nonzero(as_tuple=True)[0]
        kk = int(min(k, len(rt)))
        if kk:
            v[l, h, rt[torch.argsort(score[l, h][rt])[:kk]]] = False
    return v, kk


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
    ap.add_argument("--n_fact", type=int, default=8,
                    help="每篇插入多少条**互不相同**的事实并各问一次。"
                         "首跑单条（答案 12–16 tok）信度只有 48.7%%（抛硬币），"
                         "噪声 ∝ 1/√T ⇒ 这是最直接的放大办法。")
    ap.add_argument("--n_recv", type=int, default=4)
    ap.add_argument("--n_don", type=int, default=4)
    ap.add_argument("--ks", default="1,4,16",
                    help="每个动作转移多少个 KV 条目。**不要只用 k=1** —— "
                         "真实控制器每头挪动的量级是几十到几百条，k=1 既最接近"
                         "数值地板、又不对应任何实际决策粒度；报表会打印 |A| 随 k "
                         "的曲线，若不随 k 增长就说明触到了地板。")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interaction", action="store_true",
                    help="每个动作额外跑 2 次前向，量出交互项 "
                         "I = A − (g⁺ − g⁻)。**这是判断「可分性/势能表示」"
                         "成不成立的唯一办法** —— |I| ≪ |A| 才能把 A 写成 u_i − u_j。")
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
        ctx_ids, tgt_ids, qid, meta = make_task(
            m, ids, a.max_ctx, a.window, a.n_fact, rng)
        # ---- 三道守卫，与 `scratch_ctrl_teacher.py` 逐条对齐 ----
        if len(ctx_ids) < a.chunk // 2:
            print(f"doc{di} 太短 ({len(ctx_ids)})，跳过"); continue
        if a.ratio * len(ctx_ids) <= a.window:
            # wrapper.py:273-275 在 ratio·clen < window 时把 chunk_ratio 置 0，
            # `_threshold(·,0)` 取 thres=max ⇒ valid 恒为全 False：保留集等于
            # 局部窗口、与分数无关。此时任何配额转移都是构造性无操作。
            print(f"doc{di}: clen={len(ctx_ids)} ≤ window/ratio="
                  f"{a.window/a.ratio:.0f}，chunk_ratio 会塌缩到 0，跳过")
            continue
        ctx_t = torch.tensor([ctx_ids], device=m.device)
        t_t = torch.tensor([tgt_ids], device=m.device)
        qid_t = torch.tensor(qid, device=m.device)
        n_q = a.n_fact

        # **分数必须在驱逐发生时录下来**（与 `scratch_ctrl_teacher.py` 同一手法）。
        # 真机首跑证明：预填结束后 `kv.score` 只剩最后一段（实测 4096 = window），
        # 拿不到全长；而 `prune_chunk` 里 `torch.stack(self.score,0)[..., lo:hi]`
        # 正好是该 chunk 驱逐区间的分数，逐块拼起来与 `valid` 对齐。
        rec_s = []
        _orig_pc = RetainCache.prune_chunk

        def _pc(self, ratio, evict_range=tuple, level="pair"):
            lo, hi = evict_range
            sc_ = torch.stack(self.score, 0)[..., lo:hi]
            rec_s.append((lo, hi, (sc_[:, 0] if sc_.dim() == 4 else sc_).float().cpu()))
            return _orig_pc(self, ratio, evict_range, level)

        RetainCache.prune_chunk = _pc
        try:
            kv = m.prefill(ctx_t, prefill_chunk_size=a.chunk, do_score=True,
                           chunk_ratio=a.ratio, window_size=a.window, level=a.level)
        finally:
            RetainCache.prune_chunk = _orig_pc
        n_seen = kv._seen_tokens
        raw_valid = kv.valid                                # 形状见下
        if raw_valid is None:
            # `prune_chunk` 一次都没被调用 ⇒ 没有驱逐决策可控。教师那边对应
            # 「doc 没有触发驱逐，跳过」。不拦会在下一行 .clone() 抛 AttributeError。
            print(f"doc{di}: 没有触发驱逐（valid is None），跳过")
            del kv; torch.cuda.empty_cache(); continue
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
        # 逐块拼分数。`valid` 覆盖 [start_idx, end_idx)（真机实测 112877 =
        # end−start），而各 chunk 的 evict_range 连续铺满这个区间，所以按 lo 排序
        # 拼接后与 `valid` 逐位对齐。**若区间不连续这里会形状不符而中止，不会静默错位。**
        assert rec_s, "prune_chunk 一次都没触发"
        rec_s.sort(key=lambda x: x[0])
        sc = torch.cat([t for _, _, t in rec_s], dim=-1).to(base_valid.device)
        cov = sum(hi - lo for lo, hi, _ in rec_s)
        # 真机实测：`valid` 覆盖 [start,end) 全上下文（112877），而各 chunk 的
        # evict_range 只铺到 108781 —— 差值**恰为 4096 = window_size**。
        # 末尾这段是**永远保留的局部窗口**，从不进驱逐决策，因此
        # **不能参与转移**（拿它当施主等于动一个方法根本控制不了的集合）。
        # 处理方式：把 valid 与 score 都截到可驱逐区，并断言尾部确实全 True。
        tail = n_ev - cov
        assert tail >= 0 and cov == sum(hi - lo for lo, hi, _ in rec_s), (cov, n_ev)
        if tail:
            assert bool(base_valid[..., cov:].all()), \
                f"尾部 {tail} 列不是全保留，局部窗口假设不成立"
            assert tail == a.window or tail == a.window - kv.start_idx, \
                f"尾部长度 {tail} 既不等于 window={a.window} 也不等于 window−sink"
            base_valid = base_valid[..., :cov].contiguous()
            n_ev = cov
        assert sc.shape[-1] == n_ev, (sc.shape, n_ev)
        assert sc.shape == base_valid.shape, (
            f"score/valid 形状不匹配 sc={tuple(sc.shape)} valid={tuple(base_valid.shape)} "
            f"raw_valid={vshape} score_raw={tuple(kv.score.shape) if torch.is_tensor(kv.score) else ('list', len(kv.score), tuple(kv.score[0].shape))} "
            f"start={kv.start_idx} end={kv.end_idx} n_ev={n_ev}")
        B0 = int(base_valid.sum())

        # ---- 自检④：score 与 valid 的对齐 ----
        # `valid` 由 `prune_chunk` 逐块 `torch.cat` 而成，覆盖 [start_idx,
        # start_idx+n_ev)；若这个区间假设错了，`sc` 与 `valid` 会整体错位，
        # 而错位**不会报错**——只会让「最好的被驱逐者/最差的保留者」全选错。
        # 保留者的分数必须系统性高于被驱逐者，否则立刻中止。
        _mr = float(sc[base_valid].mean()); _me = float(sc[~base_valid].mean())
        assert _mr > _me, f"score/valid 错位：保留 {_mr:.4f} ≤ 驱逐 {_me:.4f}"
        print(f"  自检④ 保留者均分 {_mr:.4f} > 被驱逐者 {_me:.4f} ✓", flush=True)

        j0, j0q = answer_nll(m, kv, t_t, qid_t, n_q, n_seen)
        print(f"doc{di}: clen={len(ctx_ids)} 保留 {B0} "
              f"({B0/base_valid.numel():.3f})  基线 NLL {j0:.4f}  "
              f"答案 {meta['n_ans']} tok / target {meta['n_tgt']}  "
              f"事实 {meta['n_fact']} 条", flush=True)

        # ---- 自检 ①：零动作必须逐位复现基线 ----
        def _wb(vv):
            """把可驱逐区掩码写回完整形状：尾部局部窗口恒 True，batch 轴按需补。"""
            full = raw_valid.clone()
            tgt = full[:, 0] if full.dim() == 4 else full
            tgt[..., :n_ev] = vv.to(tgt.device)
            return full

        kv.valid = _wb(base_valid)
        j_null, _ = answer_nll(m, kv, t_t, qid_t, n_q, n_seen)
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
            kv.valid = _wb(v)
            jj, jjq = answer_nll(m, kv, t_t, qid_t, n_q, n_seen)
            Aq = (j0q - jjq)            # J = −NLL ⇒ A = NLL0 − NLL'，逐问句
            rec = dict(doc=di, recv=list(i), don=list(j), k=kk, tag=tag,
                       A=float(Aq.mean()), Aq=[float(x) for x in Aq],
                       A_odd=float(Aq[1::2].mean()), A_even=float(Aq[0::2].mean()))
            if a.interaction:
                # **交互项 I_ij** —— 外部评审的核心数学批评：
                #     A_{i←j} = J(S∪{i}\{j}) − J(S)
                #             = [J(S∪{i})−J(S)] + [J(S∪{i}\{j})−J(S∪{i})]
                # 第二项条件于「i 已加入」，与条件于 S 的 −g⁻_j **一般不等**，
                # 因为 softmax 分母同时被两侧改动：Z → Z + e^{qk_i} − e^{qk_j}。
                # 所以 `A ≈ g⁺−g⁻` 是**近似**，可分性必须测不能假设。
                # 两次额外前向即可量出 I：
                v_add, _ = apply_one_side(base_valid, sc, i, kk, True)
                kv.valid = _wb(v_add)
                j_add, _ = answer_nll(m, kv, t_t, qid_t, n_q, n_seen)
                v_rem, _ = apply_one_side(base_valid, sc, j, kk, False)
                kv.valid = _wb(v_rem)
                j_rem, _ = answer_nll(m, kv, t_t, qid_t, n_q, n_seen)
                gp = j0 - j_add          # 加 i 的收益（NLL 降多少）
                gm = j_rem - j0          # 减 j 的代价（NLL 升多少）
                rec.update(gp=gp, gm=gm, I=float(Aq.mean()) - (gp - gm))
            recs.append(rec)
        kv.valid = raw_valid
        del kv
        torch.cuda.empty_cache()

    if not recs:
        print("没有任何动作被评估"); return
    A = np.array([r["A"] for r in recs])
    Aq = np.array([r["Aq"] for r in recs])                    # [n_act, n_q]
    h1 = np.array([r["A_odd"] for r in recs])                 # 奇数问句
    h2 = np.array([r["A_even"] for r in recs])                # 偶数问句
    from scipy import stats as st
    print(f"\n=== {len(A)} 个动作 ===")
    print(f"  A 均值 {A.mean():+.5f}  sd {A.std():.5f}  为正 {np.mean(A>0):.1%}")
    print(f"  |A| 中位 {np.median(np.abs(A)):.5f}  最大 {np.abs(A).max():.5f}")

    # ---- 对照：|A| 是否随 k 增长。这同时是**数值噪声地板**的检验 ----
    # 掩码相同 ⇒ 前向逐位相同（自检①测到 A 恰为 0），所以没有"重复测量抖动"；
    # 但掩码不同会改变 flash-attn 的归约长度，仍有 rounding 差。若 |A| 在 k=1
    # 与 k=16 上一样大，说明测到的是地板而不是效应。
    print(f"\n=== |A| 随 k（若不增长 ⇒ 触到数值/离散地板，k=1 不可用）===")
    for kk in sorted({r["k"] for r in recs}):
        sub = np.array([r["A"] for r in recs if r["k"] == kk])
        print(f"  k={kk:<4d} n={len(sub):<4d} |A| 中位 {np.median(np.abs(sub)):.5f}"
              f"  均值 {sub.mean():+.5f}")

    # ---- 自检③：信度。**按问句奇偶交错对半**，不是按 token 位置 ----
    # 为什么不按 token 位置：前半/后半答案 token 不是同一潜在量的重复测量
    # （教师强制下后半条件于答案前缀，对缓存依赖更弱），所以那种分法的低一致率
    # 无法区分"估计量噪声大"与"真效应本来就随位置变"。按**独立事实**分则两半
    # 估的是同一个 Ā = E_q[·]，才是真正的重复测量。取奇/偶而非前/后，是为了让
    # 两半的平均问句位置相同，抵消拼接带来的顺序效应。
    print(f"\n=== 自检③ 信度（奇数问句 A vs 偶数问句 A，n_q={Aq.shape[1]}）===")
    r_half = st.pearsonr(h1, h2)[0]
    r_sb = 2 * r_half / (1 + r_half) if r_half > -1 else float("nan")
    print(f"  Pearson  {r_half:+.3f}   Spearman {st.spearmanr(h1,h2)[0]:+.3f}")
    print(f"  Spearman-Brown 校正到全量 {Aq.shape[1]} 问句: r = {r_sb:+.3f}")
    nz = (h1 != 0) & (h2 != 0)
    print(f"  符号一致率 {np.mean(np.sign(h1[nz])==np.sign(h2[nz])):.1%}（n={nz.sum()}）")

    # ---- 逐动作 bootstrap（对**问句**重采样）----
    rs = np.random.default_rng(0)
    nq = Aq.shape[1]
    bs = Aq[:, rs.integers(0, nq, size=(2000, nq))].mean(-1)   # [n_act, 2000]
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=1)
    pos, neg = (lo > 0), (hi < 0)
    print(f"\n=== 逐动作 95% bootstrap CI（重采样问句，2000 次）===")
    print(f"  有益 (CI 下界>0) {pos.sum()}/{len(A)} = {pos.mean():.1%}")
    print(f"  有害 (CI 上界<0) {neg.sum()}/{len(A)} = {neg.mean():.1%}")
    print(f"  不可分 (0∈CI)   {(~pos & ~neg).sum()}/{len(A)} = {(~pos & ~neg).mean():.1%}")
    print(f"  ⇒ 判据：**可分比例 ≥ 50% 且信度 r_sb ≥ 0.6、符号一致 ≥ 75%** 才可训练；")
    print(f"    否则标签噪声主导。注意反过来不成立 —— 低一致率只说明**这个估计量**")
    print(f"    方差过大，不说明真优势为零。")

    if a.interaction and "I" in recs[0]:
        I = np.array([r["I"] for r in recs])
        GP = np.array([r["gp"] for r in recs]); GM = np.array([r["gm"] for r in recs])
        appr = GP - GM
        print(f"\n=== 交互项 I = A − (g⁺ − g⁻)：可分性/势能表示成不成立 ===")
        print(f"  |A| 中位 {np.median(np.abs(A)):.5f}   |I| 中位 {np.median(np.abs(I)):.5f}"
              f"   **|I|/|A| 中位 {np.median(np.abs(I))/max(np.median(np.abs(A)),1e-12):.3f}**")
        print(f"  近似 (g⁺−g⁻) 与精确 A：Pearson {st.pearsonr(appr,A)[0]:+.3f}  "
              f"Spearman {st.spearmanr(appr,A)[0]:+.3f}  "
              f"符号一致 {np.mean(np.sign(appr)==np.sign(A)):.1%}")
        print(f"  ⇒ |I|/|A| 远小于 1 且符号一致率高 ⇒ 可写成势能 u_i−u_j；否则**不能**")
    hp = [r for r in recs if r["tag"] == "heur"]; rp = [r for r in recs if r["tag"] == "rand"]
    if hp and rp:
        print(f"\n=== 启发式受主 vs 随机受主（对照）===")
        print(f"  启发式 A 均值 {np.mean([r['A'] for r in hp]):+.5f} (n={len(hp)})")
        print(f"  随机   A 均值 {np.mean([r['A'] for r in rp]):+.5f} (n={len(rp)})")
    json.dump(recs, open(os.path.join(ROOT, a.out), "w"))
    print(f"\n写出 {a.out}")


if __name__ == "__main__":
    main()
