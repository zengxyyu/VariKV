#!/usr/bin/env python3
"""ProMeta Student 的训练驱动 —— **在线教师**，不落盘中间标签。

────────────────────────────────────────────────────────────────────────────
为什么在线
────────────────────────────────────────────────────────────────────────────
Student 要算 `Û = softmax(q̂·K)`，**需要那一篇的 K**。而 `[L,Hkv,16000,128]`
的 fp16 一块就 459 MB，落盘不现实。同时这条路上唯一昂贵的一步是**预填**
（一篇 ~110k token 约 1 分钟），教师的 M 次未来前向各只有几十个 token、
`future_utility` 的 einsum 是 ~1e11 FLOP —— 都可忽略。
⇒ 预填一次，教师标签与 Student 前向**共用同一份 K/V**，什么都不用存。

────────────────────────────────────────────────────────────────────────────
四条纪律（每一条都对应本仓库栽过的一类错）
────────────────────────────────────────────────────────────────────────────
① **不许报没有参照的损失。** 每步同时打印**平凡解**的损失：Student 输出
   均匀分布时 `KL(P‖unif) = log n − H(P)`。`gap` 目标那次「loss 0.003」
   其实只比 `m≡0` 好 10–15%，就是没有参照害的。
② **梯度范数必须非零**（Stage 2b 吃过「loss 在降、`|grad|max` 恒 0」）。
③ **划分不许依赖会增长的目录。** 用 `--n_docs` 固定篇数、按**位置**切
   train/val，不做 shuffle（`--split_seed` 那个坑：篇数一变划分全变）。
④ **反循环性**：每轮打印教师语料的 `demand_structure`。把语料造成「有共享
   需求」等于断言真实负载有共享需求 —— 这个断言必须能被读者查。

⚠ **一次训练不是一次测量**：报 n≥3 个种子与跨种子散布，否则什么都别报。

    .venv/bin/python -u scratch_prometa_train.py --seed 0 --epochs 3 \\
        --out varikv/prometa_s0.pt
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "external/FastKVzip/prefill"))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("-g", "--gate", default="fastkvzip")
    # ── 语料 ─────────────────────────────────────────────────────────────
    # `cat`：上游现成的 `fineweb_10k_cat` **10 篇** 103k–120k 长文（旧默认）。
    # `mix`：用 `fineweb_10k` 的 **68 篇短文**（10k–26k）**自己拼**，长度按
    #        `--len_mix` 分层抽样 ⇒ **可以造任意多条互不相同的上下文，且长短并存**。
    #
    # **为什么加它**（2026-08-22，用户指出）：FastKVzip 的门控训练集本身就是
    # **长短混合**（`feature.py:26`：`fineweb_10k` 前 29 篇 + `fineweb_10k_cat`
    # 前 5 篇 = 34 篇），**只用长文的是我，不是它**。而 `fineweb_10k_cat` 一共
    # 就 10 篇，独立上下文数被死死卡在 10。分块技术并不要求训练上下文很长 ——
    # `chunk_ranges` 对 16k 的短文给 1 段、对 110k 给 7–8 段，形状一样，
    # Student 的预测靶子恒定是 16000 个位置。**所以长度不是必需，多样性才是。**
    # ⚠ **措辞更正**：我先前说「这套机制 `scratch_adv_teacher.py` 早就有，
    #    这里只是接过来」——**不准确**。那里有的是**想法**，实现差三处，
    #    而且它的约束与我们**不同**：
    #      · 它按**字符**（`min_chars`）拼，我按 **token** 目标长度**分层抽样**；
    #      · 它**只造长上下文**，因为它的教师是「反事实配额优势」，**必须让驱逐
    #        非退化**（`clen > window/ρ = 40,960`）；**ProMeta 的教师是满缓存
    #        注意力需求，根本不驱逐 ⇒ 这条约束对我们不成立**，这正是我们可以
    #        长短并存的原因；
    #      · 它「拼到够长就停、取前缀」，我改成「拼到目标+裕量后随机开窗 + 去重」。
    #    ⚠ 第三处**不是修它的 bug，是修我自己的**：它只造 ≥40,960 的长文，
    #    必然拼 3–4 篇，前缀碰撞基本不可能；是我引入短目标（8k/16k < 单篇 10k–29k）
    #    才让「取前缀」退化成「截断同一篇」，实测 64 条里只有 60 条互不相同。
    #
    # ⚠ **必须随数字一起写的限制**：拼接买到的是**组合多样性**，不是**新内容**。
    #    独立底文的天花板仍是 `fineweb_10k` 的 **68 篇**（外加 `_cat` 的 10 篇，
    #    而那 10 篇本身也是同一族语料的拼接）。要真正突破这个天花板，只能换语料
    #    （RestoreKV 用 LongAlpaca 500 篇 + PG-19 50 本 + Tulu-3 FLAN 1500 例）。
    p.add_argument("--corpus", default="fineweb_10k_cat",
                   choices=["fineweb_10k_cat", "mix", "pool"])
    p.add_argument("--pool_band", default="10000,30000",
                   help="`pool` 取哪个长度 band（token）。10k–30k 有 4,328 篇可用")
    p.add_argument("--pool_skip", type=int, default=68,
                   help="与 load_fineweb 返回的 68 篇完全不相交（cat 那 5 篇是"
                        "同 band 前 ~40 篇拼的，只跳 34 仍会重叠）")
    p.add_argument("--len_mix", default="8000:0.25,16000:0.35,32000:0.25,110000:0.15",
                   help="`mix` 的长度分层：`目标token:权重` 逗号分隔。"
                        "默认照 RestoreKV/LookaheadKV 的做法以短为主、保留 15% 长样本")
    p.add_argument("--corpus_seed", type=int, default=20260822)
    p.add_argument("--n_docs", type=int, default=10,
                   help="**固定**篇数。划分按位置切，不 shuffle")
    p.add_argument("--n_val", type=int, default=2)
    p.add_argument("--max_ctx", type=int, default=120000)
    p.add_argument("--chunk", type=int, default=16000)
    p.add_argument("--window", type=int, default=4096)
    p.add_argument("--n_chunk", type=int, default=3,
                   help="每篇取几个 chunk 训练（等距覆盖首/中/尾）")
    p.add_argument("--n_fact", type=int, default=3)
    p.add_argument("--n_joint", type=int, default=2)
    # **默认 Q-only**（外部复核指出，采纳）：`qa` 的答案里含 `TAG=<value>`，
    # 而 `<value>` 与插进上下文的事实**字符串完全相同** ⇒ 教师 query 直接命中
    # 目标位置，会强化合成检索捷径。Q+A 作为消融，不作主结果。
    p.add_argument("--span", default="q", choices=["q", "qa"])
    p.add_argument("--n_probe", type=int, default=0,
                   help="Student 的 probe 数；0 = 跟随教师的 M")
    p.add_argument("--d_proj", type=int, default=128)
    p.add_argument("--n_pool", type=int, default=4)
    p.add_argument("--d_lat", type=int, default=64)
    p.add_argument("--pool_layer", type=int, default=14)
    p.add_argument("--lam_div", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_context", action="store_true",
                   help="**上下文致盲对照**：把池化摘要置零 ⇒ probe 变成常数。"
                        "与完整 Student 的差 = 上下文路径的全部贡献。"
                        "不可分 ⇒ 当前数据上「从上下文预测未来」没有发生。")
    p.add_argument("--val_seed", type=int, default=987654,
                   help="**验证任务的 rng，不含 epoch** ⇒ 跨 epoch 是同一把尺子")
    p.add_argument("--out", default="varikv/prometa_s0.pt")
    p.add_argument("--ablate_span", action="store_true",
                   help="同一篇上同时抽 q 与 qa 两套标签并报一致性，然后退出")
    return p.parse_args()


def student_demand(net, kv, lo, hi, pool_end, pool_layer, no_context=False):
    """Student 在 chunk `[lo,hi)` 上的需求分布 `Û: [Ms,L,Hkv,n]`（带梯度）。

    **因果口径与部署逐字一致**：部署时 `prune_chunk((lo,hi))` 被调用的那一刻，
    cache 里正好有 `[0, hi+window)`（上游 `end_idx = len(score) − window`），
    所以摘要只能看到 `pool_end = hi + window`。这里由调用方算好传进来。
    """
    if no_context:
        # **上下文致盲对照**（2026-08-22 加）：把池化摘要置零 ⇒
        # `u = trunk(0) + probe_bias` 与上下文**无关**，Student 退化成一组
        # **常数** probe。它与完整 Student 的差，就是「上下文路径」的全部贡献。
        #
        # 为什么这条必须先跑：Student 的 302,464 个参数里，**上下文相关的那条
        # 路径（proj + pool_q + trunk）占 279,616 个，而它的输出只有
        # `u ∈ R^{5×64} = 320` 个数**。训练集只有 8 篇 × 5 chunk = **40 个不同
        # 的池化输入**（跨 epoch 只重抽注入的事实，10 万 token 的底文不变，
        # 池化摘要由底文主导）⇒ 40×320 = 12,800 个目标数对 279,616 个参数，
        # **严重过参数化**。若致盲对照与完整 Student 不可分，那么
        # 「从上下文预测未来需求」这件事在当前数据上**根本没有发生**，
        # 此时扩语料是唯一正确的动作；若可分，才说明上下文路径确有信号。
        z = torch.zeros(net.pool_q.shape[0], net.proj.out_features,
                        device=net.pool_q.device, dtype=net.probe_bias.dtype)
        q = net.from_pooled(z)
    else:
        V = kv.value_cache[pool_layer][0][:, :pool_end, :]      # [Hkv,n,d]
        flat = V.permute(1, 0, 2).reshape(pool_end, -1).to(net.proj.weight.dtype)
        q = net.from_pooled(net.pool(flat))                     # [Ms,L,Hkv,d]
    Ms, L, H, d = q.shape
    out = []
    for l in range(L):
        K = kv.key_cache[l][0][:, lo:hi, :].to(q.dtype)
        out.append(torch.softmax(
            torch.einsum("mhd,hnd->mhn", q[:, l], K) / d ** 0.5, dim=-1))
    return torch.stack(out, 1), q                               # [Ms,L,H,n]


def build_corpus(a, enc):
    """→ `n_docs` 条**token id 列表**。`cat` 用现成长文；`mix` 用 68 篇短文自己拼。

    划分仍按**位置**切（`docs[:n_tr]` 训练 / 其余验证），与 `--corpus_seed` 无关 ⇒
    换 `--seed` 不会改动划分（`--split_seed` 那个坑：篇数一变划分全变）。
    """
    from data.load import load_fineweb
    if a.corpus == "pool":
        # **绕开 1M 上限**：独立文档的真天花板是 4,328 篇（见 teacher.load_fineweb_pool）
        from prometa.teacher import load_fineweb_pool
        lo_, hi_ = (int(x) for x in a.pool_band.split(","))
        txt = load_fineweb_pool(a.n_docs, lo_, hi_, skip=a.pool_skip)
        out = [enc(t) for t in txt]
        L = [len(x) for x in out]
        print(f"[corpus] pool：{len(out)} 篇**真正独立**的文档（band "
              f"[{lo_:,},{hi_:,})，跳过前 {a.pool_skip} 篇），token "
              f"{min(L):,}-{max(L):,} 合计 {sum(L):,}", flush=True)
        return out
    if a.corpus != "mix":
        return [enc(d["context"]) for d in load_fineweb(a.corpus)][:a.n_docs]
    shorts = [enc(d["context"]) for d in load_fineweb("fineweb_10k")]
    pairs = [x.split(":") for x in a.len_mix.split(",")]
    tgts = [int(t) for t, _ in pairs]
    wts = np.array([float(w) for _, w in pairs], dtype=float)
    wts = wts / wts.sum()
    out, used, seen = [], set(), set()
    for i in range(a.n_docs):
        # ⚠ **必须留出裕量再随机开窗，不能总取前缀**（2026-08-22 干跑抓到）：
        # 目标 8k/16k **小于一篇底文**（fineweb_10k 是 10k–26k），若像
        # `scratch_adv_teacher.mix` 那样「拼到 >= tgt 就停、再取前缀」，
        # 抽到同一篇首文的两条上下文就**逐位完全相同** —— 实测 64 条里只有
        # 60 条互不相同。改成拼到 `tgt + slack` 后在缓冲区内**随机开窗**，
        # 并对整条上下文做**去重硬闸**（重复就换种子重抽，最多 20 次）。
        for attempt in range(20):
            r = random.Random((a.corpus_seed, i, attempt, "mix").__hash__())
            tgt = tgts[int(np.searchsorted(np.cumsum(wts), r.random()))]
            order = list(range(len(shorts)))
            r.shuffle(order)
            buf, picked = [], []
            for j in order:
                buf += shorts[j]; picked.append(j)
                if len(buf) >= tgt + max(2000, tgt // 4):
                    break
            off = r.randrange(0, max(1, len(buf) - tgt + 1))
            ctx = buf[off:off + tgt]
            key = hash(tuple(ctx[:64]) + tuple(ctx[-64:]) + (len(ctx),))
            if key not in seen:
                break
        assert key not in seen, f"第 {i} 条重抽 20 次仍与已有上下文重复"
        seen.add(key)
        used.update(picked)
        out.append(ctx)
    L = [len(x) for x in out]
    assert len(seen) == len(out), f"去重硬闸失效：{len(seen)} != {len(out)}"
    print(f"[corpus] mix：{len(out)} 条**互不相同**，长度 {min(L):,}-{max(L):,}（中位 "
          f"{int(np.median(L)):,}）；**独立底文只有 {len(shorts)} 篇，用到 {len(used)} 篇** "
          f"⇒ 拼接买到的是组合多样性、不是新内容", flush=True)
    return out


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    from data.load import load_fineweb
    from model import ModelKVzip
    from prometa.model import ProMetaPredictor, diversity_loss
    from prometa.teacher import (build_task, chunk_ranges, demand_structure,
                                 extract_U)
    from prometa.train import match_loss, to_dist

    model = ModelKVzip(a.model, "retain", a.gate)
    L = model.config.num_hidden_layers
    Hkv = model.config.num_key_value_heads
    dh = model.config.hidden_size // model.config.num_attention_heads
    sys_len = int(model.sys_prompt_ids.shape[1])
    enc = lambda s: model.encode(s)[0].tolist()

    Mt = a.n_fact + a.n_joint
    Ms = a.n_probe or Mt
    assert Ms >= Mt, (Ms, Mt)
    net = ProMetaPredictor(Hkv * dh, dh, L, Hkv, n_future=Ms, d_proj=a.d_proj,
                           n_pool=a.n_pool, d_lat=a.d_lat).to(model.device).float()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    npar = sum(p.numel() for p in net.parameters())
    print(f"[train] L={L} Hkv={Hkv} d={dh} sys_len={sys_len} | "
          f"Mt={Mt} Ms={Ms} | Student {npar:,} 参数 | seed={a.seed}", flush=True)

    docs = build_corpus(a, enc)
    assert len(docs) == a.n_docs, f"语料只有 {len(docs)} 篇，要 {a.n_docs}"
    n_tr = a.n_docs - a.n_val
    print(f"[train] 语料 {a.corpus} 固定 {a.n_docs} 篇（{sum(len(d) for d in docs):,} token），**按位置**切 "
          f"train[0:{n_tr}] / val[{n_tr}:{a.n_docs}]（不 shuffle）", flush=True)

    def one_doc(di, epoch, train=True):
        """→ (demand_val, trivial_val, n_used, ds_stats)

        ⚠ **验证的 rng 里不许有 `epoch`**（2026-08-22，外部复核指出，采纳）。
        留出**文档**（doc 8,9）确实从未训练过，但若验证任务每个 epoch 重新抽
        key/value/插入位置/格式模板，跨 epoch 的 val 曲线**就不是同一把尺子**，
        不能用来说「没有过拟合」。训练侧保留 `epoch`（那是有益的数据增广），
        **验证侧固定**。
        """
        rng = (random.Random((a.seed, epoch, di).__hash__()) if train
               else random.Random((a.val_seed, di).__hash__()))
        ids = docs[di]                      # 已是 token id 列表
        ctx, futures, meta = build_task(enc, ids, a.max_ctx, a.window,
                                        a.n_fact, rng, n_joint=a.n_joint)
        assert len(futures) == Mt, (len(futures), Mt)
        n_total = sys_len + len(ctx)
        _, usable = chunk_ranges(n_total, sys_len, a.chunk, a.window)
        if not usable:
            print(f"  doc{di}: 没有可用 chunk（clen={len(ctx)}），跳过")
            return None
        pick = [usable[i] for i in np.unique(np.linspace(
            0, len(usable) - 1, min(a.n_chunk, len(usable))).round().astype(int))]
        U_by, kv, info = extract_U(model, ctx, futures, pick, span=a.span,
                                   prefill_chunk=a.chunk, verbose=False)

        ds = demand_structure(U_by[pick[len(pick) // 2]])
        tot, triv, cnt, dvs = 0.0, 0.0, 0, 0.0
        for lo, hi in pick:
            Us = to_dist(torch.as_tensor(U_by[(lo, hi)], device=model.device))
            pool_end = min(hi + a.window, info["n_prefix"])
            if train:
                Uh, q = student_demand(net, kv, lo, hi, pool_end, a.pool_layer,
                                       no_context=a.no_context)
                # **demand 与 div 必须分开记**（外部复核指出，采纳）：训练侧
                # 之前打印的是 `demand + λ·div`，验证侧只有 `demand`，两者被
                # 直接并排比较且都拿「平凡解」当参照 —— 而平凡解只是 `demand`
                # 的基线。现在两边都只用 `demand` 做学习曲线，`div` 单独报。
                dem = match_loss(Us, Uh)
                dv = diversity_loss(q)
                loss = dem + a.lam_div * dv
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                # 纪律②：本仓库吃过「loss 在降、grad 恒零」的亏
                # 致盲对照下 proj/pool_q **本就该**零梯度，所以只要求总范数非零
                assert float(gn) > 0, "梯度范数为 0 —— 图断了，训练是假的"
                if a.no_context and _step0[0]:
                    _dead = [n_ for n_, p_ in net.named_parameters()
                             if p_.grad is None or float(p_.grad.abs().max()) == 0]
                    print(f"  [no_context] 零梯度参数（应恰为 proj/pool_q）：{_dead}",
                          flush=True)
                    assert set(_dead) == {"proj.weight", "pool_q"}, _dead
                    _step0[0] = False
                opt.step()
            else:
                with torch.no_grad():
                    Uh, q = student_demand(net, kv, lo, hi, pool_end, a.pool_layer,
                                           no_context=a.no_context)
                    dem = match_loss(Us, Uh)
                    dv = diversity_loss(q)
                gn = torch.tensor(0.0)
            # 纪律①：平凡解参照 —— Student 输出均匀分布时的损失
            n = hi - lo
            with torch.no_grad():
                triv_l = float((np.log(n) + (Us * torch.log(
                    Us.clamp_min(1e-12))).sum(-1)).mean())
            tot += float(dem); triv += triv_l; dvs += float(dv); cnt += 1
            print(f"  doc{di} chunk[{lo},{hi}) n={n} pool_end={pool_end} "
                  f"demand={float(dem):.4f} 平凡解={triv_l:.4f} "
                  f"(相对降低 {1 - float(dem)/max(triv_l,1e-9):+.1%}) "
                  f"div={float(dv):.4f} |g|={float(gn):.3e}", flush=True)
            del Us, Uh, q, dem, dv
        del U_by, kv
        torch.cuda.empty_cache()
        return tot / cnt, triv / cnt, cnt, ds, dvs / cnt

    if a.ablate_span:
        # **Q-only vs Q+A**：教师用真答案是否构成泄漏，用数字回答，不用措辞
        from prometa.risk import topb_mask
        rng = random.Random((a.seed, 0, 0).__hash__())
        ctx, futures, _ = build_task(enc, docs[0], a.max_ctx, a.window,
                                     a.n_fact, rng, n_joint=a.n_joint)
        _, usable = chunk_ranges(sys_len + len(ctx), sys_len, a.chunk, a.window)
        pick = [usable[len(usable) // 2]]
        outs = {}
        for sp in ("q", "qa"):
            U, kv, _i = extract_U(model, ctx, futures, pick, span=sp,
                                  prefill_chunk=a.chunk, verbose=False)
            outs[sp] = U[pick[0]]
            del kv; torch.cuda.empty_cache()
        Uq, Uqa = outs["q"], outs["qa"]
        n = Uq.shape[-1]
        for rho in (0.02, 0.05, 0.1, 0.2):
            k = max(1, int(round(rho * n)))
            A = topb_mask(Uq.max(0), k); B = topb_mask(Uqa.max(0), k)
            J = float(((A & B).sum(-1) / np.maximum((A | B).sum(-1), 1)).mean())
            print(f"[ablate span] rho={rho:<5} k={k:<6} J(Q-only, Q+A) = {J:.4f}")
        print(f"[ablate span] 需求结构  Q-only {demand_structure(Uq)}")
        print(f"[ablate span] 需求结构  Q+A    {demand_structure(Uqa)}")
        print("⇒ J 接近 1 ⇒ 真答案没带来额外信息，应当直接用 Q-only（泄漏质疑消失）；"
              "\n  J 明显 < 1 ⇒ 两者不是同一个标签，必须分别训练并各报一次下游。")
        print("Finished.")
        return

    _step0 = [True]
    t0 = time.time()
    for ep in range(a.epochs):
        tr = [one_doc(di, ep, True) for di in range(n_tr)]
        tr = [x for x in tr if x]
        va = [one_doc(di, ep, False) for di in range(n_tr, a.n_docs)]
        va = [x for x in va if x]
        f = lambda xs, i: float(np.mean([x[i] for x in xs])) if xs else float("nan")
        # 学习曲线**只看 demand**；div 单独报（它与平凡解不可比）
        ds = tr[0][3] if tr else {}
        print(f"[epoch {ep}] train demand {f(tr,0):.4f} (平凡 {f(tr,1):.4f}, "
              f"相对 {1-f(tr,0)/max(f(tr,1),1e-9):+.1%}) | "
              f"**val demand {f(va,0):.4f} (平凡 {f(va,1):.4f}, "
              f"相对 {1-f(va,0)/max(f(va,1),1e-9):+.1%})** | "
              f"div train {f(tr,4):.4f} val {f(va,4):.4f} | "
              f"教师需求结构 J(mean,max)={ds.get('J_mean_max',float('nan')):.4f} "
              f"shared={ds.get('shared_frac',float('nan')):.4f} "
              f"conc_only_mean={ds.get('conc_only_mean',float('nan')):.4f} "
              f"conc_only_max={ds.get('conc_only_max',float('nan')):.4f} | "
              f"{time.time()-t0:.0f}s", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()},
                        args=vars(a), epoch=ep,
                        arch=dict(hidden_dim=Hkv * dh, head_dim=dh, n_layers=L,
                                  n_kv_heads=Hkv, n_future=Ms, d_proj=a.d_proj,
                                  n_pool=a.n_pool, d_lat=a.d_lat),
                        model=a.model, teacher=dict(corpus=a.corpus, span=a.span,
                                                    n_fact=a.n_fact,
                                                    n_joint=a.n_joint)), a.out)
        print(f"[epoch {ep}] saved {a.out}", flush=True)
    print("⚠ 一次训练不是一次测量：至少 3 个种子、报跨种子散布。")
    print("Finished.")


if __name__ == "__main__":
    main()
