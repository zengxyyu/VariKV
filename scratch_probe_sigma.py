#!/usr/bin/env python3
"""Δs 的尺度到底该用逐头 σ 还是全局 σ —— 这是 `ControlMemory.delta` 的一个未验证假设。

`delta` 现在返回

    Δs_i = α · σ_h(l,h) · tanh(head(...)),      σ_h = std over the head's candidates

但 `level="pair"` 的阈值 τ 是**跨层跨头全局**的一个标量（`score.py:_threshold`：
把所有 (L,H,n) 展平排序取第 int(N·ratio)−1 个）。也就是说，决定去留的是 `s0 − τ`
这个**原始单位**的量，而控制器的输出被缩放到了**逐头单位**。

两者不一致会产生一个具体后果：若某个头的分数分布很窄（σ_h ≪ σ_global），那么
α·σ_h 就远小于它到 τ 的距离，**这个头的候选无论控制器输出什么都翻不过阈值**；
反过来宽分布的头会独占全部控制权。而 trainer 的 global 损失恰恰在要求跨头重排，
架构却可能表达不了 —— 这正是"优化目标与参数化不匹配"的典型形态。

本探针只回答两个数，不做任何训练：

  1. `σ_h / σ_global` 的分布（跨 112 个 (层,kv头) 组）。若跨度只有 2–3 倍，
     问题是理论上的；若跨越一两个数量级，就是实打实的表达力缺陷。
  2. **可达性**：在 α=α_max 的满功率下，每组有多少候选满足 |s0−τ| < α·σ_h，
     即真的有机会被翻转。这个数按组统计，报中位数与为 0 的组占比。

判读：若"可达候选数为 0 的组"占比高，就必须把 Δs 改成两分量
（逐头 σ 管头内重排 + 全局 σ 管跨头预算搬移），或至少把 log(σ_h/σ_g) 喂进头。
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from data.load import load_fineweb                               # noqa: E402
from model import ModelKVzip                                     # noqa: E402
from attention.kvcache import RetainCache                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--gate", default="fastkvzip")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--chunk", type=int, default=16000)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--level", default="pair")
    ap.add_argument("--max_ctx", type=int, default=32768)
    ap.add_argument("--n_docs", type=int, default=2)
    ap.add_argument("--alpha_max", type=float, default=1.0)
    a = ap.parse_args()

    m = ModelKVzip(a.model, "retain", a.gate)
    docs = [d["context"] for d in load_fineweb("fineweb_10k_cat")[:a.n_docs]]

    for di, txt in enumerate(docs):
        ids = m.encode(txt)[0].tolist()[-a.max_ctx:]
        chunks = []
        orig = RetainCache.prune_chunk

        def rec(self, ratio, evict_range, level="pair"):
            lo, hi = evict_range
            s0 = torch.stack(self.score, 0)[..., lo:hi].clone()
            th, r_ = orig(self, ratio, evict_range, level)
            chunks.append((s0[:, 0].float().cpu(), float(th)))
            return th, r_

        RetainCache.prune_chunk = rec
        try:
            kv = m.prefill(torch.tensor([ids], device=m.device),
                           prefill_chunk_size=a.chunk, do_score=True,
                           chunk_ratio=a.ratio, window_size=a.window, level=a.level)
        finally:
            RetainCache.prune_chunk = orig
        del kv
        if not chunks:
            print(f"doc{di} ({len(ids)} tok) 未触发驱逐"); continue

        for ci, (s0, thr) in enumerate(chunks):            # s0 [L,H,n]
            L, H, n = s0.shape
            sg = float(s0.reshape(-1).std())
            sh = s0.std(dim=-1)                            # [L,H]
            rat = (sh / sg).reshape(-1).numpy()
            d = (s0 - thr).abs()                            # [L,H,n]
            # **"可争夺"候选**：落在 τ 的 ±0.5σ_g 内，即有现实可能被翻的那些。
            # 不先筛出它们，"不可达"会和"这组本来就整体远离 τ"混在一起——
            # 后者是全局阈值化的正常现象（整头整层被成批保留或丢弃），不是缺陷。
            cont = (d < 0.5 * sg)                           # [L,H,n]
            reach = cont & (d < a.alpha_max * sh[..., None])
            nc = cont.sum(-1).reshape(-1).numpy()
            nr = reach.sum(-1).reshape(-1).numpy()
            has = nc > 0
            cov = nr[has] / np.maximum(nc[has], 1)          # 组内可达率
            q = lambda v, p: float(np.percentile(v, p))     # noqa: E731
            pc = np.percentile(s0.reshape(-1).numpy(), [1, 50, 99])
            print(f"doc{di} chunk{ci}  n={n}  σ_g={sg:.4g}  τ={thr:.4g}  "
                  f"s0 p1/p50/p99 = {pc[0]:.3g}/{pc[1]:.3g}/{pc[2]:.3g}")
            print(f"   σ_h/σ_g   min {rat.min():.3f}  p10 {q(rat,10):.3f}  "
                  f"med {q(rat,50):.3f}  p90 {q(rat,90):.3f}  max {rat.max():.3f}"
                  f"   （跨度 {rat.max()/max(rat.min(),1e-12):.0f}×）")
            print(f"   有可争夺候选的组 {has.mean():.1%}  "
                  f"其中可达率 med {q(cov,50):.1%}  p10 {q(cov,10):.1%}  "
                  f"**可达率<10% 的组占比 {float((cov < 0.1).mean()):.1%}**", flush=True)


if __name__ == "__main__":
    sys.exit(main())
