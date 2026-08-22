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
from collections import Counter as _Counter

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
    # ★ **正式路径**：直接读 `prometa/dataset.py` 产出的 manifest。
    # 给了它就**完全忽略** --corpus / --n_docs / --n_fact 等内部造数据的参数，
    # 划分也用 manifest 里的 `split` 字段（源文档级、已验证互不相交），
    # 不再按位置切。⚠ 这是「正式版正在用新数据集训练」的唯一判据。
    p.add_argument("--drop_kinds", default="",
                   help="逗号分隔，从 manifest 里剔除这些 kind。默认剔除 continuation "
                        "以保证 M 一致——见 --allow_mixed_M 的说明")
    p.add_argument("--allow_mixed_M", action="store_true",
                   help="允许 manifest 里各条记录的未来数不同（Ms>Mt）。**默认关**")
    p.add_argument("--manifest", default="",
                   help="prometa_data/manifest_v1.jsonl。给了就走正式数据路径")
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
    p.add_argument("--n_atoms", type=int, default=8,
                   help="每个未来的检索 atom 数。教师是 max_{t,g} 的包络，Student 用 "
                        "max_r 与之同族。R=8 由 scratch_prometa_arch.py 定："
                        "真实 level=pair 全局掩码 J@0.1 R=1 0.6414 → R=8 0.7204 → "
                        "R=16 0.7360（R=16 只多 +0.016 却参数翻倍）")
    p.add_argument("--lam_atom", type=float, default=0.1,
                   help="atom 多样性权重。硬 max 下没赢过的 atom 梯度恒零会静默死掉")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle_mode", default="exact", choices=["exact", "crop"],
                   help="exact=只在**同宽**之间换（实测覆盖 62.6%，但每一次换都是"
                        "纯粹的『同一个量、换个文档』单变量交换）；crop=配不上就从"
                        "更宽捐赠者裁剪并重新归一化（覆盖 ~100%，但被裁的那 37% "
                        "同时改变了『来源文档』与『支撑截断』两件事）。**默认 exact**")
    p.add_argument("--shuffle_labels", action="store_true",
                   help="**未来打乱阴性对照**（外部复核提出，采纳）：给上下文 C_i 配"
                        "**别的文档**的教师标签 U*_j。若 loss 几乎不变 ⇒ **教师靶子"
                        "根本没和上下文绑定**。这比致盲对照更直接：致盲问「Student "
                        "有没有用上下文」，打乱问「上下文里到底有没有可用的信息」。")
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
    from prometa.model import ProMetaPredictor
    Ms, R, L, H, d = q.shape
    out, use = [], []
    for l in range(L):
        K = kv.key_cache[l][0][:, lo:hi, :].to(q.dtype)
        u, w = ProMetaPredictor.demand_layer(q[:, :, l], K, ret_usage=True)
        out.append(u); use.append(w)
    # 第三个返回值是 **atom 利用率**：硬 max 下没赢过的 atom 梯度恒零，
    # 会静默死掉；不打印它就等于没有运行时证据（本仓库铁律）。
    return torch.stack(out, 1), q, torch.stack(use, 0).mean(0)   # [Ms,L,H,n]


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
    from prometa.model import (ProMetaPredictor, diversity_loss,
                               atom_diversity_loss)
    from prometa.teacher import (build_task, chunk_ranges, demand_structure,
                                 extract_U)
    from prometa.train import match_loss, to_dist

    model = ModelKVzip(a.model, "retain", a.gate)
    L = model.config.num_hidden_layers
    Hkv = model.config.num_key_value_heads
    dh = model.config.hidden_size // model.config.num_attention_heads
    sys_len = int(model.sys_prompt_ids.shape[1])
    enc = lambda s: model.encode(s)[0].tolist()

    # ⚠ **Ms 必须从数据的唯一真源派生**（2026-08-22 审计发现，第④类错）。
    # 原写法 `Mt = a.n_fact + a.n_joint` 是**内造语料**的参数，与 manifest 无关；
    # 它现在等于 5 纯属与 manifest 的 max M 撞上。若某次 manifest 只有 M=3，
    # Ms 仍是 5 ⇒ 两个 probe 从头到尾没被任何真实未来监督过，却参与推理决策。
    _MANI_PRE = None
    if a.manifest:
        import json as _json
        _MANI_PRE = [r for r in (_json.loads(l) for l in open(a.manifest)) if r["futures"]]
        if a.drop_kinds:
            _dk = set(x.strip() for x in a.drop_kinds.split(",") if x.strip())
            _n0 = len(_MANI_PRE)
            _MANI_PRE = [r for r in _MANI_PRE if r["kind"] not in _dk]
            print(f"[train] --drop_kinds {sorted(_dk)}：{_n0} → {len(_MANI_PRE)} 条", flush=True)
        # ⚠ **混合 M 默认拒绝**（2026-08-22 外部复核指出，采纳）。原先
        # `allow_extra=(MANI is not None)` 是**无条件开**的，于是 M=1 的
        # continuation 与 M=5 的其余记录混在一起**静默通过**。问题不在代码能不能跑：
        # Mt=1 时 `match_loss` 只监督 argmin 那一个 probe，另外 4 个在**这条上下文上**
        # 没有任何真实未来约束，而推理时 `ρ_β` 会把 5 个 probe 全算进驱逐决策。
        # 「别的记录会监督到全部 5 个」只保证**参数级**覆盖，不保证**逐上下文**覆盖。
        _Ms_set = sorted(set(len(r["futures"]) for r in _MANI_PRE))
        assert a.allow_mixed_M or len(_Ms_set) == 1, (
            f"manifest 的未来数不一致：{_Ms_set}。Mt<Ms 的记录上，未被匹配的 probe "
            f"在该上下文没有任何监督却参与推理决策。用 --drop_kinds continuation "
            f"去掉 M=1 的那类，或显式 --allow_mixed_M 并在结论里写明这条边界。")
        Mt = max(len(r["futures"]) for r in _MANI_PRE)
        _mh = _Counter(len(r["futures"]) for r in _MANI_PRE)
        print(f"[train] Mt 由 manifest 派生：max M = {Mt}（分布 {dict(sorted(_mh.items()))}）",
              flush=True)
    else:
        Mt = a.n_fact + a.n_joint
    Ms = a.n_probe or Mt
    assert Ms >= Mt, (Ms, Mt)
    net = ProMetaPredictor(Hkv * dh, dh, L, Hkv, n_future=Ms, d_proj=a.d_proj,
                           n_pool=a.n_pool, d_lat=a.d_lat,
                           n_atoms=a.n_atoms).to(model.device).float()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    npar = sum(p.numel() for p in net.parameters())
    print(f"[train] L={L} Hkv={Hkv} d={dh} sys_len={sys_len} | "
          f"Mt={Mt} Ms={Ms} | Student {npar:,} 参数 | seed={a.seed}", flush=True)

    MANI = _MANI_PRE          # **同一份对象**，不重读（读两遍就有两份可能不一致的真源）
    if a.manifest:
        docs = MANI
        _C = _Counter
        _c = _C((r["split"], r["kind"]) for r in MANI)
        _m = _C(len(r["futures"]) for r in MANI)
        print(f"[train] **走正式 manifest 路径** {a.manifest}：{len(MANI)} 条可用"
              f"（有未来的），组成 {dict(_c)}，每条未来数 {dict(sorted(_m.items()))}",
              flush=True)
        assert MANI, "manifest 里没有一条带未来的记录（selfstudy 还没生成？）"
        # ⚠ **span 一致性硬闸**（2026-08-22 审计发现）。manifest 里只有 synth 的
        # future 带答案（250/920），selfstudy 与 continuation 的 `a` 是空的。
        # 于是 `--span qa` 会让 synth 用「问+答」当教师查询、其余用「只问」——
        # **同一个数据集里两套标签定义**，跨 kind 的损失不可加。静默发生、
        # 日志里看不出来（第⑤类错：理论声明与实现不一致）。宁可拒绝启动。
        _na = sum(1 for r in MANI for f in r["futures"] if f["a"])
        _nq = sum(len(r["futures"]) for r in MANI)
        assert a.span == "q" or _na in (0, _nq), (
            f"--span {a.span} 但 manifest 里只有 {_na}/{_nq} 个 future 带答案 ⇒ "
            f"标签定义会随 kind 而变。要么 --span q，要么把所有 future 补齐答案。")
        print(f"[train] span={a.span}；future 带答案 {_na}/{_nq}"
              f"（{'一致' if a.span=='q' or _na in (0,_nq) else '不一致'}）", flush=True)
    else:
        docs = build_corpus(a, enc)
        assert len(docs) == a.n_docs, f"语料只有 {len(docs)} 篇，要 {a.n_docs}"
    n_tr = a.n_docs - a.n_val
    if MANI is not None:
        # ⚠ manifest 路径下 `docs` 是**记录 dict**，`len(d)` 数的是键数不是 token；
        # `n_tr`/`a.n_docs` 更是内造语料的量，与 manifest 无关（划分见下方 `split` 字段）。
        # 旧版把这两样直接打进日志 ⇒ 一条**由代码而非数据生成的判词**，已修。
        _nt = sum(len(r["ctx"]) for r in MANI)
        _bt = _C(r["band"] for r in MANI)
        print(f"[train] manifest 语料 {len(MANI)} 条（{_nt:,} token，band {dict(sorted(_bt.items()))}），"
              f"划分**由记录自带的 `split` 字段决定**，与 --n_docs/--n_val 无关", flush=True)
    else:
        print(f"[train] 语料 {a.corpus} 固定 {a.n_docs} 篇（{sum(len(d) for d in docs):,} token），**按位置**切 "
              f"train[0:{n_tr}] / val[{n_tr}:{a.n_docs}]（不 shuffle）", flush=True)

    def one_doc(di, epoch, train=True):   # noqa: C901
        """→ (demand_val, trivial_val, n_used, ds_stats)

        ⚠ **验证的 rng 里不许有 `epoch`**（2026-08-22，外部复核指出，采纳）。
        留出**文档**（doc 8,9）确实从未训练过，但若验证任务每个 epoch 重新抽
        key/value/插入位置/格式模板，跨 epoch 的 val 曲线**就不是同一把尺子**，
        不能用来说「没有过拟合」。训练侧保留 `epoch`（那是有益的数据增广），
        **验证侧固定**。
        """
        rng = (random.Random((a.seed, epoch, di).__hash__()) if train
               else random.Random((a.val_seed, di).__hash__()))
        if MANI is not None:
            # ★ manifest 路径：上下文与未来都是**固定的**（不再每 epoch 重抽事实）。
            # 这本身就更干净 —— 数据集冻结、可复现、可挂 sha256。
            rec = docs[di]
            ctx = rec["ctx"]
            futures = [dict(q=f["q"], a=f["a"], kind=f["kind"],
                            needs=f.get("needs", [])) for f in rec["futures"]]
            meta = dict(kind=rec["kind"], band=rec["band"], M=len(futures))
        else:
            ids = docs[di]                  # 已是 token id 列表
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
        # ⚠ **只在训练侧打乱，验证侧永远用真标签**（2026-08-22 审计改，原为两侧都打乱）。
        #    两侧都打乱测的是「打乱后的学习问题好不好解」——常数预测器与真正的
        #    逐文档学习器**都会显示改善**，区分不了，等于白跑。真正的判据是
        #    「**用打乱标签训练出来的模型，还能不能预测真实需求**」：
        #      仍 +13% ⇒ 标签内容无所谓 ⇒ 那 13% 是与文档无关的通用结构；
        #      掉到 ~0 ⇒ 训练确实依赖这一篇自己的标签。
        #    附带修掉一条 val→train 信息流：原来验证文档的 U 也会进标签库、
        #    再被发给训练文档当「打乱标签」。
        if a.shuffle_labels and train:
            # **未来打乱阴性对照**：把本篇的教师标签换成**另一篇文档**的。
            # ⚠ 只换标签，上下文与 Student 的输入**一字不动** ⇒ 单变量。
            #
            # 两个实现要点（首版两处都写错了，记下来）：
            # ① 必须换**别的文档**的标签，不是「同一篇上一个 epoch」的 —— 后者
            #    只是换了一组随机注入的事实，仍然来自同一篇上下文，测不到东西；
            # ② 跨文档的 chunk 宽度不一定相同（首块 11,876、整块 16,000、末块残），
            #    形状对不上就会**静默不换** ⇒ 跑出来是普通训练却冒充成对照。
            #    所以按**宽度**建库，并**硬性断言这一篇至少换成功一次**。
            # ③（2026-08-22 修，真机崩在这里）首版写成
            #    `_label_bank[w] = (di, U_by[r].copy() if swapped == 0 else src[1])`
            #    有**两个**错：(a) 某宽度**首次出现**（`src is None`）而本篇更早的
            #    chunk 已换过时，走 else 分支取 `None[1]` ⇒ TypeError 直接崩；
            #    (b) `swapped != 0` 时存进库的是**换完之后**的标签（别人的），
            #    于是标签会在文档间接力传递，第三篇拿到的仍是第一篇的 ⇒ 对照被稀释。
            #    正确做法：**先留本篇原始标签**，换与不换都只把原始的存进库。
            # ④ 内存：一条 U 是 [M,L,H,w]，w=16000 时 fp32 约 36 MB。按宽度建库、
            #    末块宽度几乎篇篇不同 ⇒ 200 篇能堆到 GB 级。加 FIFO 上限，
            #    且**重复插入会刷新到队尾** ⇒ 占绝大多数的 16000 宽永远在库里。
            # ⑤ **宽度结构性配不上**（2026-08-22 真机冒烟抓到，崩在 epoch1）。
            #    单 chunk 文档的宽度 = `n_ctx − 4096`，**篇篇唯一**；实测全量
            #    训练集只有 62.6% 的 chunk 落在共同宽度上（11876 首块 / 16000 中块），
            #    其余 37.4% 永远配不上 ⇒ 旧的 per-doc 硬断言必崩，而放宽成
            #    「配不上就不换」又会把对照稀释成 63% 强度且**不可见**。
            #    改成：同宽**精确换**；否则从**更宽**的捐赠者裁到目标宽度并
            #    重新归一化（裁出来仍是合法分布、且确实来自别的文档）；
            #    三个计数全部打进日志，断言下放到 **epoch 级**。
            for r in pick:
                w = U_by[r].shape[-1]
                orig = U_by[r]                     # ← 换之前先留住，换完就取不到了
                # ⚠ **默认只做同宽精确交换**（2026-08-22 外部复核指出，采纳）。
                #    裁剪兜底能把覆盖率从 62.6% 抬到 ~100%，但被裁的那 37% 同时
                #    改变了两件事（来源文档 **和** 支撑截断）——尾部质量大的捐赠者
                #    被裁+重归一化后会向平凡解靠，那时 val 变差可能是截断伪影而不是
                #    「标签换错了」。一个 62.6% 但**每次都干净**的置换，比一个 100%
                #    里混着 37% 双变量改动的置换更有说服力（对照一次只变一个变量）。
                # ⚠ 首版写成 `dw >= w and dw >= lo_w`（`lo_w = w`）—— **两个条件等价，
                #    exact 模式完全没生效**，真机冒烟打出「精确同宽 0、裁剪自更宽 2」
                #    才看出来。这就是「每加一个 mode 必须同时加运行时日志」的用处。
                cand = [(dw, d_, u_) for dw, lst in _label_bank.items() for d_, u_ in lst
                        if d_ != di and (dw == w if a.shuffle_mode == "exact" else dw >= w)]
                if cand:
                    dw, _d, u = min(cand, key=lambda t: t[0])   # 最接近的更宽者
                    if dw == w:
                        U_by[r] = u; _shuf_cnt["exact"] += 1
                    else:
                        c = u[..., :w].astype("float64")
                        U_by[r] = (c / np.maximum(c.sum(-1, keepdims=True), 1e-30)
                                   ).astype(u.dtype)
                        _shuf_cnt["crop"] += 1
                else:
                    _shuf_cnt["none"] += 1
                lst = [t for t in _label_bank.pop(w, []) if t[0] != di]
                lst.append((di, orig.copy()))
                _label_bank[w] = lst[-2:]          # 每宽最多留 2 个不同来源
                while len(_label_bank) > 8:
                    _label_bank.pop(next(iter(_label_bank)))

        ds = demand_structure(U_by[pick[len(pick) // 2]])
        tot, triv, cnt, dvs = 0.0, 0.0, 0, 0.0
        _use_acc = []
        for lo, hi in pick:
            Us = to_dist(torch.as_tensor(U_by[(lo, hi)], device=model.device))
            pool_end = min(hi + a.window, info["n_prefix"])
            if train:
                Uh, q, use = student_demand(net, kv, lo, hi, pool_end, a.pool_layer,
                                            no_context=a.no_context)
                # **demand 与 div 必须分开记**（外部复核指出，采纳）：训练侧
                # 之前打印的是 `demand + λ·div`，验证侧只有 `demand`，两者被
                # 直接并排比较且都拿「平凡解」当参照 —— 而平凡解只是 `demand`
                # 的基线。现在两边都只用 `demand` 做学习曲线，`div` 单独报。
                # `allow_extra` 只在 manifest 路径开：那里 M 随记录而变
                # （continuation 只有 1 个未来、synth/selfstudy 有 5 个）。
                # ⚠ 外部复核原本反对 `Ms>Mt`，理由是「本步没被监督的 probe 在推理
                #    时仍参与决策」。**在混合 M 的数据上这条反对被化解**：所有 5 个
                #    probe 在**别的记录**上都会被监督到，只是不在每一步。
                #    若数据里全是 M=1，就又会退化成原来的问题 —— 所以下面打印
                #    每条未来数的直方图，保证这一点可查。
                dem = match_loss(Us, Uh, allow_extra=a.allow_mixed_M)
                dv = diversity_loss(q)
                da = atom_diversity_loss(q)
                loss = dem + a.lam_div * dv + a.lam_atom * da
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
                    # ⚠ 首版写成 {"proj.weight","pool_q"}，**漏了 `trunk.0.weight`** ——
                    # `z=0` 时 `trunk[0]` 算的是 `W0@0 + b0`，而线性层权重的梯度是
                    # `grad_out ⊗ input`，input 全零 ⇒ `W0` 的梯度也恒为 0。
                    # 这条断言把整个致盲对照跑崩了（0 个 epoch 完成）。
                    # `trunk.0.bias` 与 `trunk.2.*` 仍有梯度，所以只多这一个。
                    assert set(_dead) == {"proj.weight", "pool_q", "trunk.0.weight"}, _dead
                    _step0[0] = False
                opt.step()
            else:
                with torch.no_grad():
                    Uh, q, use = student_demand(net, kv, lo, hi, pool_end, a.pool_layer,
                                                no_context=a.no_context)
                    dem = match_loss(Us, Uh, allow_extra=a.allow_mixed_M)
                    dv = diversity_loss(q)
                    da = atom_diversity_loss(q)
                gn = torch.tensor(0.0)
            # 纪律①：平凡解参照 —— Student 输出均匀分布时的损失
            n = hi - lo
            with torch.no_grad():
                triv_l = float((np.log(n) + (Us * torch.log(
                    Us.clamp_min(1e-12))).sum(-1)).mean())
            tot += float(dem); triv += triv_l; dvs += float(dv); cnt += 1
            _use_acc.append(use.detach().cpu().numpy())
            print(f"  doc{di} chunk[{lo},{hi}) n={n} pool_end={pool_end} "
                  f"demand={float(dem):.4f} 平凡解={triv_l:.4f} "
                  f"(相对降低 {1 - float(dem)/max(triv_l,1e-9):+.1%}) "
                  f"div={float(dv):.4f} atomdiv={float(da):.4f} "
                  f"atom用量={'/'.join(f'{float(x):.2f}' for x in use)} "
                  f"|g|={float(gn):.3e}", flush=True)
            del Us, Uh, q, dem, dv, use
        del U_by, kv
        torch.cuda.empty_cache()
        return (tot / cnt, triv / cnt, cnt, ds, dvs / cnt,
                np.mean(_use_acc, 0) if _use_acc else np.zeros(1))

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
    # 打乱对照的标签库：**按 chunk 宽度**索引 → (来源文档号, U 数组)
    _label_bank = {}
    _shuf_cnt = {"exact": 0, "crop": 0, "none": 0}
    t0 = time.time()
    if MANI is not None:
        idx_tr = [i for i, r in enumerate(MANI) if r["split"] == "train"]
        idx_va = [i for i, r in enumerate(MANI) if r["split"] == "val"]
        print(f"[train] 划分来自 manifest 的 `split` 字段（**源文档级、已验证互不相交**）："
              f"train {len(idx_tr)} / val {len(idx_va)}", flush=True)
    else:
        idx_tr, idx_va = list(range(n_tr)), list(range(n_tr, a.n_docs))
    for ep in range(a.epochs):
        # **训练顺序按 (seed, epoch) 打乱**（2026-08-22 审计加）。逐 chunk 就是一步
        # SGD，固定顺序既是一个跨种子共有的混淆，又让「跨种子散布」只反映初始化
        # 方差、系统性低估真实方差。验证侧顺序**不打乱**（顺序不影响均值，且要可比）。
        _ord = list(idx_tr)
        random.Random((a.seed, ep, 0x5eed).__hash__()).shuffle(_ord)
        for _k in _shuf_cnt: _shuf_cnt[_k] = 0
        tr = [one_doc(di, ep, True) for di in _ord]
        if a.shuffle_labels:
            _sw = _shuf_cnt["exact"] + _shuf_cnt["crop"]
            _tt = _sw + _shuf_cnt["none"]
            print(f"  [shuffle] epoch{ep}：换掉 {_sw}/{_tt} = {_sw/max(_tt,1):.1%} 个 chunk 的标签"
                  f"（精确同宽 {_shuf_cnt['exact']}、裁剪自更宽 {_shuf_cnt['crop']}、"
                  f"无捐赠者 {_shuf_cnt['none']}）", flush=True)
            _floor = 0.5 if a.shuffle_mode == "exact" else 0.9
            assert ep == 0 or _sw / max(_tt, 1) >= _floor, (
                f"打乱对照只换掉 {_sw}/{_tt} ⇒ 低于 {_floor:.0%}，说明不了问题。"
                f"（exact 模式的结构上限约 62.6%，见 DATASET.md 的宽度分布）")
        tr = [x for x in tr if x]
        va = [one_doc(di, ep, False) for di in idx_va]
        va = [x for x in va if x]
        f = lambda xs, i: float(np.mean([x[i] for x in xs])) if xs else float("nan")
        # 学习曲线**只看 demand**；div 单独报（它与平凡解不可比）
        ds = tr[0][3] if tr else {}
        print(f"[epoch {ep}] train demand {f(tr,0):.4f} (平凡 {f(tr,1):.4f}, "
              f"相对 {1-f(tr,0)/max(f(tr,1),1e-9):+.1%}) | "
              f"**val demand {f(va,0):.4f} (平凡 {f(va,1):.4f}, "
              f"相对 {1-f(va,0)/max(f(va,1),1e-9):+.1%})** | "
              f"div train {f(tr,4):.4f} val {f(va,4):.4f} | "
              # **atom 死亡是静默的** —— 最小用量掉到 ~0 就说明那个 atom 从没赢过
              # 任何位置、梯度恒零，R=8 已经退化。必须逐 epoch 可查。
              f"**atom用量 {'/'.join(f'{x:.3f}' for x in (np.mean([t[5] for t in tr], 0) if tr else [0]))}"
              f"（最小 {min(np.mean([t[5] for t in tr], 0)) if tr else 0:.3f}）** | "
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
                                  n_pool=a.n_pool, d_lat=a.d_lat,
                                  n_atoms=a.n_atoms),
                        model=a.model, teacher=dict(corpus=a.corpus, span=a.span,
                                                    n_fact=a.n_fact,
                                                    n_joint=a.n_joint)), a.out)
        print(f"[epoch {ep}] saved {a.out}", flush=True)
    print("⚠ 一次训练不是一次测量：至少 3 个种子、报跨种子散布。")
    print("Finished.")


if __name__ == "__main__":
    main()
