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
    p.add_argument("--corpus", default="fineweb_10k_cat")
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
    p.add_argument("--val_seed", type=int, default=987654,
                   help="**验证任务的 rng，不含 epoch** ⇒ 跨 epoch 是同一把尺子")
    p.add_argument("--out", default="varikv/prometa_s0.pt")
    p.add_argument("--ablate_span", action="store_true",
                   help="同一篇上同时抽 q 与 qa 两套标签并报一致性，然后退出")
    return p.parse_args()


def student_demand(net, kv, lo, hi, pool_end, pool_layer):
    """Student 在 chunk `[lo,hi)` 上的需求分布 `Û: [Ms,L,Hkv,n]`（带梯度）。

    **因果口径与部署逐字一致**：部署时 `prune_chunk((lo,hi))` 被调用的那一刻，
    cache 里正好有 `[0, hi+window)`（上游 `end_idx = len(score) − window`），
    所以摘要只能看到 `pool_end = hi + window`。这里由调用方算好传进来。
    """
    V = kv.value_cache[pool_layer][0][:, :pool_end, :]          # [Hkv,n,d]
    flat = V.permute(1, 0, 2).reshape(pool_end, -1).to(net.proj.weight.dtype)
    q = net.from_pooled(net.pool(flat))                         # [Ms,L,Hkv,d]
    Ms, L, H, d = q.shape
    out = []
    for l in range(L):
        K = kv.key_cache[l][0][:, lo:hi, :].to(q.dtype)
        out.append(torch.softmax(
            torch.einsum("mhd,hnd->mhn", q[:, l], K) / d ** 0.5, dim=-1))
    return torch.stack(out, 1), q                               # [Ms,L,H,n]


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

    docs = [d["context"] for d in load_fineweb(a.corpus)][:a.n_docs]
    assert len(docs) == a.n_docs, f"语料只有 {len(docs)} 篇，要 {a.n_docs}"
    n_tr = a.n_docs - a.n_val
    print(f"[train] 语料 {a.corpus} 固定 {a.n_docs} 篇，**按位置**切 "
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
        ids = enc(docs[di])
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
                Uh, q = student_demand(net, kv, lo, hi, pool_end, a.pool_layer)
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
                assert float(gn) > 0, "梯度范数为 0 —— 图断了，训练是假的"
                opt.step()
            else:
                with torch.no_grad():
                    Uh, q = student_demand(net, kv, lo, hi, pool_end, a.pool_layer)
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
        ctx, futures, _ = build_task(enc, enc(docs[0]), a.max_ctx, a.window,
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
