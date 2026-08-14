#!/usr/bin/env python3
"""VariKV-B 最终版的**未来效用教师**：离线为每个候选 KV 算 U_i。

要学的量有精确定义（见 `attention/control_memory.py` 顶部）：无历史选择器最好只能预测
`E[U|X]`，有历史是 `E[U|X,M]`，控制器学的是二者之差。所以先得有 U。

--------------------------------------------------------------------------------
U 的推导（用精确式，不是一阶近似）

单个 query 位置 t、query-head g，注意力输出 `o = Σ_j p_j v_j`。把 token i 拿掉后
softmax 重新归一：

    o^{-i} = (o − p_i v_i)/(1 − p_i)
    Δo^{(i)} = o^{-i} − o = [p_i/(1−p_i)] · (o − v_i)

GPT 草案里的 `a_qi·(v_i − o_q)` 是它在 p_i→0 时的一阶版本。精确因子免费，而且对
高注意力 token（恰恰是决定成败的那些）差别很大，所以这里用精确式。

**GQA 聚合**：token i 属于 kv-head h，同时影响共享它的 G 个 query-head。我们自己的
P0 探针测过**逐头损伤有 75% 在层内自相消**（`‖Σ_h W_O Δo_h‖/Σ_h‖W_O Δo_h‖ = 0.253`），
所以必须先经 `W_O` 求和再取模，不能逐头取模再相加：

    U_i = mean_t ‖ Σ_{g∈group(h)} W_O^{(g)} · [p/(1−p)]_{t,i}^{(g)} · (o_t^{(g)} − v_i^{(h)}) ‖

**为什么不物化那个向量**：[T, n, d_model] = 256×1024×3584 = 940M 浮点。改用 Gram：
`‖Wz‖² = zᵀΓz`，`Γ = WᵀW` 是 (G·d)×(G·d) = 896×896，比投影到 3584 维便宜 4 倍。

--------------------------------------------------------------------------------
两处**记录在案的近似**（不是 bug，是取舍）

1. softmax 分母只对**上下文键**求和，不含 target 块内部的因果自注意力。
   省掉的项对每个 query 是一个公共常数，会把该 query 的所有 p_i 同比例缩放；
   由于 U 在 query 上取均值，这确实会轻微改变 query 之间的相对权重。
   上下文 32768 vs target ≤256，量级很小。
2. "未来 query" 用文档自身的续写 token。真实评测里 query 是检索问题，几何未必相同——
   这正是 CLAUDE.md 记录的「不匹配的是任务，不是长度」。第一版沿用现有训练语料的做法。
"""
import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
sys.path.insert(0, ROOT)
os.chdir(os.path.join(ROOT, "external/FastKVzip/prefill"))

from attention.kvcache import RetainCache                    # noqa: E402
from data.load import load_fineweb                           # noqa: E402
from model.wrapper import ModelKVzip                         # noqa: E402

_Q = {}
_orig_prepare = RetainCache.prepare


def _patched_prepare(self, q, k, v, l):
    """抓 post-RoPE 的 query（与 scratch_probe_cluster.py:76 同一套已验证的钩子）。"""
    _Q[l] = q.detach()
    return _orig_prepare(self, q, k, v, l)


RetainCache.prepare = _patched_prepare


@torch.no_grad()
def utility(m, kv, tgt_ids, layers, cand_idx, n_qpos=16):
    """→ {layer: U [H, n_cand]}，U 是移除该候选造成的平均 |ΔW_O·o|。

    cand_idx: {layer: LongTensor [H, n_cand]}，缓存坐标下的候选位置。
    """
    d = kv.head_dim if hasattr(kv, "head_dim") else m.config.hidden_size // m.config.num_attention_heads
    H = m.config.num_key_value_heads
    # **必须先记下上下文长度再前向**。`m.model(...)` 走的是 HF 模型而不是 wrapper 的
    # `__call__`，所以没有 `kv.slice()` 回滚 —— target 的 K/V 会被追加进 cache 且留在
    # 那里。若在前向之后直接读 `kv.key_cache`，拿到的是 context+target，而我们手写的
    # softmax 没有因果掩码 ⇒ 位置 t 会看到 t+1 之后的 target key。
    n_ctx = kv.key_cache[layers[0]].shape[2]
    _Q.clear()
    m.model(tgt_ids, past_key_values=kv)          # 只为触发钩子拿到 query
    out = {}
    for l in layers:
        Aq = _Q[l][0].float() * (d ** -0.5)        # [HQ,T,d]，已含 RoPE
        HQ, T, _ = Aq.shape
        G = HQ // H
        # target 位置抽样：T 全用会让下面的 [T,n,G,d] 爆掉，且相邻位置高度相关
        tsel = torch.linspace(0, T - 1, min(n_qpos, T)).long().to(Aq.device)
        Aq = Aq.view(H, G, T, d)[:, :, tsel]       # [H,G,Tq,d]
        WO = m.model.model.layers[l].self_attn.o_proj.weight.detach().float()  # [dm, HQ*d]
        K = kv.key_cache[l][0][:, :n_ctx].float()  # **切到上下文**，排除 target 键
        V = kv.value_cache[l][0][:, :n_ctx].float()
        idx = cand_idx[l]                          # [H,n_cand]
        U = torch.zeros(H, idx.shape[1], device=Aq.device)
        for h in range(H):
            g0 = h * G
            # Γ = WᵀW，W = 该组 G 个头的 o_proj 列块拼接 [dm, G*d]
            W = WO[:, g0 * d:(g0 + G) * d]
            Gram = W.T @ W                          # [G*d, G*d]
            S = torch.einsum("gtd,nd->gtn", Aq[h], K[h])          # [G,Tq,n_ctx]
            P = S.softmax(-1)
            o = torch.einsum("gtn,nd->gtd", P, V[h])              # [G,Tq,d]
            p_i = P[..., idx[h]]                                  # [G,Tq,n_cand]
            w = p_i / (1.0 - p_i).clamp_min(1e-4)                 # 精确移除因子
            v_i = V[h][idx[h]]                                    # [n_cand,d]
            # z[t,n,g,:] = w * (o − v_i)；[Tq,n,G,d]，Tq=16/n=1024/G=7/d=128 ⇒ 118 MB
            z = w.permute(1, 2, 0)[..., None] * (
                o.permute(1, 0, 2)[:, None] - v_i[None, :, None, :])
            Tq, nc = z.shape[0], z.shape[1]
            zf = z.reshape(Tq * nc, G * d)
            U[h] = ((zf @ Gram) * zf).sum(-1).clamp_min(0).sqrt().view(Tq, nc).mean(0)
            del S, P, o, p_i, w, z, zf
        out[l] = U
        del Aq, K, V
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--gate", default="fastkvzip")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--chunk", type=int, default=16000)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--level", default="pair")
    ap.add_argument("--max_ctx", type=int, default=32768)
    ap.add_argument("--target_len", type=int, default=256)
    ap.add_argument("--n_short", type=int, default=29)
    ap.add_argument("--n_long", type=int, default=5)
    ap.add_argument("--n_qpos", type=int, default=16)
    ap.add_argument("--n_keep", type=int, default=256,
                    help="每 (chunk,层,kv头) 留多少个**最靠近阈值**的候选做排序损失。"
                         "离阈值远的无论 Δs 多大都翻不了——手工版实测 β=0.5 只翻 0.895%")
    ap.add_argument("--n_rand", type=int, default=512,
                    help="额外随机采样，供 writer 汇总 retained/evicted 用")
    ap.add_argument("--out", default="scratch_ctrl_traces")
    a = ap.parse_args()

    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    m = ModelKVzip(a.model, "retain", a.gate)
    L = m.config.num_hidden_layers
    H = m.config.num_key_value_heads
    layers = list(range(L))

    n_prune = []
    docs = (load_fineweb("fineweb_10k")[:a.n_short]
            + load_fineweb("fineweb_10k_cat")[:a.n_long])
    print(f"训练文档 {len(docs)} 篇", flush=True)

    for di, s in enumerate(docs):
        f = os.path.join(ROOT, a.out, f"doc{di:03d}.pt")
        if os.path.exists(f):
            print(f"跳过 doc{di}"); continue
        ids = s.ids if hasattr(s, "ids") else s
        need = a.max_ctx + a.target_len
        ctx_ids, tgt = ids[-need:-a.target_len], ids[-a.target_len:]
        if len(ctx_ids) < a.chunk // 2:
            print(f"doc{di} 太短 ({len(ctx_ids)})，跳过"); continue
        ctx_t = torch.tensor([ctx_ids], device=m.device)
        tgt_t = torch.tensor([tgt], device=m.device)

        # ---- 基线预填，逐 chunk 记录 (score0, valid, thres) ----
        chunks = []
        orig = RetainCache.prune_chunk

        def rec(self, ratio, evict_range, level="pair"):
            lo, hi = evict_range
            s0 = torch.stack(self.score, 0)[..., lo:hi].clone()
            th, r_ = orig(self, ratio, evict_range, level)
            chunks.append(dict(lo=lo, hi=hi, thres=th,
                               s0=s0[:, 0].float().cpu(),
                               valid=self.valid[..., -(hi - lo):].clone().cpu()))
            return th, r_

        RetainCache.prune_chunk = rec
        try:
            kv_p = m.prefill(ctx_t, prefill_chunk_size=a.chunk, do_score=True,
                             chunk_ratio=a.ratio, window_size=a.window,
                             level=a.level)
        finally:
            RetainCache.prune_chunk = orig
        if not chunks:
            print(f"doc{di} 没有触发驱逐，跳过"); continue
        n_prune.append(len(chunks))
        # **q 必须来自满缓存轨迹。** 上面那次预填带压缩掩码，target 前向时 attention
        # 只看保留下来的 KV ⇒ 得到的 query 是「压缩轨迹的 query」。而 U 要回答的是
        # 「若满缓存可用，删掉这个 token 会损失多少」，所以 (q, K, V) 必须同属满缓存
        # 轨迹，否则是个混合反事实——正是之前"局部探针与全局轨迹混用"栽过的坑。
        # 物理 K/V 两次预填相同（RetainCache 不真删），差别只在 q。
        del kv_p
        torch.cuda.empty_cache()
        kv = m.prefill(ctx_t, prefill_chunk_size=a.chunk, do_score=False)

        # ---- 每个 chunk 抽样候选，算 U ----
        rec_out = []
        for c in chunks:
            lo, hi, n = c["lo"], c["hi"], c["hi"] - c["lo"]
            cand, kept = {}, {}
            for l in layers:
                dist = (c["s0"][l] - c["thres"]).abs()             # [H,n]
                near = dist.argsort(dim=-1)[:, :a.n_keep]
                g = torch.Generator().manual_seed(1000 * di + lo + l)
                rnd = torch.stack([torch.randperm(n, generator=g)[:a.n_rand]
                                   for _ in range(H)])
                sel = torch.cat([near, rnd], dim=-1)               # [H,n_keep+n_rand]
                cand[l] = (sel + lo).to(m.device)
                kept[l] = sel
            U = utility(m, kv, tgt_t, layers, cand, n_qpos=a.n_qpos)
            per_l = []
            for l in layers:
                sel = kept[l]
                pos = (sel + lo).to(m.device)
                kk = torch.stack([kv.key_cache[l][0][h, pos[h]] for h in range(H)])
                vv = torch.stack([kv.value_cache[l][0][h, pos[h]] for h in range(H)])
                per_l.append(dict(
                    k=kk.half().cpu(), v=vv.half().cpu(),
                    s0=torch.gather(c["s0"][l], 1, sel).cpu(),
                    ret=torch.gather(c["valid"][l], 1, sel).cpu(),
                    U=U[l].float().cpu(), n_near=a.n_keep))
            rec_out.append(dict(thres=c["thres"], layers=per_l))
        torch.save(dict(doc=di, chunks=rec_out, H=H, L=L), f)
        print(f"doc{di}: {len(rec_out)} 次驱逐 → {f}", flush=True)
        del kv
        torch.cuda.empty_cache()


    # **序列深度必须报出来。** B 学的是 M_{t-1} → selection_t：若每篇只有 1 次驱逐，
    # 记忆永远没机会影响任何决策，训练是空转的。CLAUDE.md 记过
    # max_ctx=32768/chunk=16000 实测只有 1.03 次 prune —— 所以这个统计是硬性检查。
    from collections import Counter
    print(f"\n【每篇文档的驱逐次数分布】{dict(sorted(Counter(n_prune).items()))}")
    if n_prune and max(n_prune) <= 1:
        print("  ⚠ 全部只有 1 次驱逐 ⇒ 记忆从未参与任何决策，这批 trace 不能用来训 B。"
              "\n     提高 --max_ctx（cat 文档有 103k–122k）或降低 --chunk。")


if __name__ == "__main__":
    sys.exit(main())
