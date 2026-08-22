#!/usr/bin/env python3
"""ProMeta 的 **GPU 冒烟** —— 在真机真 cache 上验管道，不验质量。

**为什么这一条排在所有 ProMeta 评测之前**：本仓库的地板那条线，66 格网格
在一个哨兵值上整块空跑、冒充成方法结果，直到几小时后才被发现（第⑫类错）。
唯一能防住这一类的，是**构造性零点对照**：把方法的强度旋钮拧到 0，断言输出
与基线**逐位相同**。地板线用 `b_min=0`（`_fbm00` 实测逐样本 Δ 全零），
ProMeta 用 `gamma=0`。

跑五件事（任一失败即 `SystemExit`）：

  A. `prometa.teacher.chunk_ranges` 与真机 `prune_chunk` 收到的 `evict_range`
     **逐项相同**。它是上游循环的第二份实现，不对拍就是第④类错。
  B. **`gamma=0` 逐位等同基线**：`valid` 逐位相同、`score` 逐位相同、
     同一段探针 token 上的 logits 逐位相同。
  C. `gamma>0` 时掩码**确实变了**（否则整条通路是死的），且预算守恒。
  D. **逐头配额没有塌向均匀** —— 这是 2026-08-22 那个撤回的真机复核：
     首版 `_z(s0)` 混合会把配额 CV 压掉、把饿死头全救活（＝混进地板效应）。
  E. 在线池化（部署路径）与离线池化（训练路径）在**真 V** 上数值等价。

    .venv/bin/python scratch_prometa_smoke.py --ratio 0.1
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "external/FastKVzip/prefill"))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("-g", "--gate", default="fastkvzip")
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--chunk", type=int, default=16000)
    p.add_argument("--window", type=int, default=4096)
    p.add_argument("--doc", type=int, default=0)
    p.add_argument("--max_ctx", type=int, default=120000)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--pool_layer", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def fail(msg):
    print(f"\n✗ FAIL: {msg}", flush=True)
    raise SystemExit(1)


def main():
    a = parse()
    torch.manual_seed(a.seed)
    from data.load import load_fineweb
    from attention.kvcache import RetainCache
    from model import ModelKVzip
    from prometa.cache import make_prometa_cache
    from prometa.model import ProMetaPredictor
    from prometa.pool import OnlineAttnPool
    from prometa.teacher import chunk_ranges

    model = ModelKVzip(a.model, "retain", a.gate)
    L = model.config.num_hidden_layers
    Hkv = model.config.num_key_value_heads
    dh = model.config.hidden_size // model.config.num_attention_heads
    sys_len = int(model.sys_prompt_ids.shape[1])
    print(f"[smoke] L={L} Hkv={Hkv} head_dim={dh} sys_len={sys_len}", flush=True)

    txt = load_fineweb("fineweb_10k_cat")[a.doc]["context"]
    ids = model.encode(txt)[:, -a.max_ctx:].to(model.device)
    clen = int(ids.shape[1])
    if a.ratio * clen <= a.window:
        fail(f"clen={clen} ≤ window/ratio={a.window/a.ratio:.0f} ⇒ chunk_ratio 塌成 0，"
             f"整条通路构造性无操作，这个配置测不出任何东西")
    print(f"[smoke] ctx {clen} tokens（非退化：clen > window/ratio = "
          f"{a.window/a.ratio:.0f}）", flush=True)

    # 未训练的 Student —— 冒烟只验管道，**不验质量**。
    net = ProMetaPredictor(Hkv * dh, dh, L, Hkv, n_future=5,
                           d_proj=128, n_pool=4, d_lat=64).to(model.device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    print(f"[smoke] Student 参数 {sum(p.numel() for p in net.parameters()):,}"
          f"（**未训练**，本脚本只验管道）", flush=True)

    # ── 抓真机 evict_range ────────────────────────────────────────────────
    seen = []
    _orig = RetainCache.prune_chunk

    def _rec(self, ratio, evict_range=tuple, level="pair"):
        seen.append(tuple(evict_range))
        return _orig(self, ratio, evict_range, level)

    def run(pm=None):
        seen.clear()
        RetainCache.prune_chunk = _rec
        try:
            kv = model._init_kv(evict_range=(sys_len, sys_len + clen))
            if pm is not None:
                kv.__class__ = pm["cls"]
                kv.pm_init(net, beta=a.beta, gamma=pm["gamma"],
                           combine=pm["combine"], pool_layer=a.pool_layer,
                           verbose=True)
            # 走上游 prefill；`_init_kv(kv=...)` 会把已建好的 cache 原样返回
            # 实例级临时覆盖，用完从 __dict__ 里删掉（赋回去会留下永久实例属性）
            model._init_kv = lambda kv_=None, evict_range=(0, 0): kv
            try:
                with torch.inference_mode():
                    out = model._prefill_impl(
                        ids, prefill_chunk_size=a.chunk, do_score=False,
                        window_size=a.window, chunk_ratio=a.ratio, level="pair")
            finally:
                model.__dict__.pop("_init_kv", None)
            probe = model.encode("\nQuestion: summarise.\nAnswer:").to(model.device)
            with torch.inference_mode():
                lg = model(probe, out, update_cache=False,
                           return_logits=True).logits[0].float().cpu().clone()
            sc = torch.stack([s.float().cpu().clone() for s in out.score], 0)
            vd = out.valid.cpu().clone()
            st = list(getattr(out, "pm_stats", []))
            rg = list(seen)
            del out
            torch.cuda.empty_cache()
            return dict(valid=vd, score=sc, logits=lg, stats=st, ranges=rg)
        finally:
            RetainCache.prune_chunk = _orig

    PMCls = make_prometa_cache(RetainCache)

    print("\n" + "=" * 72 + "\n[A/B] 基线 RetainCache", flush=True)
    base = run(None)
    print("\n" + "=" * 72 +
          "\n[B] ProMeta gamma=0（构造性零点，必须逐位等同基线）", flush=True)
    z0 = run(dict(cls=PMCls, gamma=0.0, combine="resid"))
    print("\n" + "=" * 72 +
          f"\n[C/D] ProMeta gamma={a.gamma} resid", flush=True)
    on = run(dict(cls=PMCls, gamma=a.gamma, combine="resid"))
    print("\n" + "=" * 72 + "\n[D] ProMeta replace", flush=True)
    rp = run(dict(cls=PMCls, gamma=1.0, combine="replace"))

    print("\n" + "=" * 72 + "\n判定\n" + "=" * 72, flush=True)

    # ── A. chunk_ranges 对拍 ─────────────────────────────────────────────
    allr, use = chunk_ranges(sys_len + clen, sys_len, a.chunk, a.window)
    if list(base["ranges"]) != [tuple(x) for x in allr]:
        fail(f"A chunk_ranges 与真机不符\n  真机 {base['ranges'][:4]} …（{len(base['ranges'])} 段）"
             f"\n  预测 {allr[:4]} …（{len(allr)} 段）")
    print(f"A ✓ chunk_ranges 与真机 {len(allr)} 段 evict_range 逐项相同"
          f"（首 {allr[0]}、末 {allr[-1]}；usable {len(use)}）")

    # ── B. gamma=0 逐位等同 ───────────────────────────────────────────────
    dv = int((base["valid"] != z0["valid"]).sum())
    ds = float((base["score"] - z0["score"]).abs().max())
    dl = float((base["logits"] - z0["logits"]).abs().max())
    if dv or ds or dl:
        fail(f"B gamma=0 不等同基线：valid 差 {dv} 位、score max|Δ|={ds:.3e}、"
             f"logits max|Δ|={dl:.3e}")
    if z0["stats"]:
        fail("B gamma=0 竟然进了 ProMeta 分支（pm_stats 非空）—— 短路失效")
    print(f"B ✓ gamma=0 与基线**逐位相同**：valid 差 0 位 / "
          f"score max|Δ|=0.0 / logits max|Δ|=0.0，且未进 ProMeta 分支")

    # ── C. gamma>0 确实动了，且预算守恒 ──────────────────────────────────
    diff = int((base["valid"] != on["valid"]).sum())
    kb, ko = int(base["valid"].sum()), int(on["valid"].sum())
    rel = abs(ko - kb) / max(kb, 1)
    if diff == 0:
        fail("C gamma>0 掩码与基线完全相同 ⇒ 整条 ProMeta 通路是死的")
    if not on["stats"]:
        fail("C pm_stats 为空 ⇒ prune_chunk 覆写没被调用")
    if rel > 1e-3:
        fail(f"C 预算不守恒：基线保留 {kb}、ProMeta {ko}（相对差 {rel:.2%}）")
    Js = [s["J"] for s in on["stats"]]
    print(f"C ✓ gamma={a.gamma} 改动 {diff} 位（占 {diff/base['valid'].numel():.3%}）；"
          f"预算 {kb} → {ko}（相对差 {rel:.2e}）；"
          f"逐 chunk J(base,prometa) 均值 {np.mean(Js):.4f} "
          f"[{min(Js):.4f},{max(Js):.4f}]，{len(Js)} 个 chunk")

    # ── D. 逐头配额没有塌向均匀（撤回项的真机复核）───────────────────────
    def quota(v):
        x = v[:, 0] if v.dim() == 4 else v                # [L,H,n]
        return x.reshape(-1, x.shape[-1]).float().sum(-1)
    qb, qo, qr = quota(base["valid"]), quota(on["valid"]), quota(rp["valid"])
    cv = lambda q: float(q.std() / q.mean().clamp_min(1e-9))
    z = lambda q: int((q == 0).sum())
    print(f"D   逐头配额 CV：基线 {cv(qb):.4f} → resid {cv(qo):.4f} → "
          f"replace {cv(qr):.4f}")
    print(f"D   饿死头（配额 0）：基线 {z(qb)} → resid {z(qo)} → replace {z(qr)}"
          f"（共 {qb.numel()} 个头）")
    if cv(qo) < 0.5 * cv(qb):
        fail(f"D resid 把配额压向均匀（CV {cv(qb):.3f} → {cv(qo):.3f}）"
             f"⇒ 混进了「向均匀收缩」这个已知的大效应，读数不可用")
    if z(qb) > 0 and z(qo) == 0:
        fail(f"D resid 把 {z(qb)} 个饿死头全救活了 ⇒ 与地板干预不可区分")
    print(f"D ✓ 配额结构保住（CV 未塌、饿死头未被系统性救活）")

    # ── E. 在线池化 ≡ 离线池化（真 V）───────────────────────────────────
    with torch.inference_mode():
        kv = model._init_kv(evict_range=(sys_len, sys_len + clen))
        model._init_kv = lambda kv_=None, evict_range=(0, 0): kv
        try:
            model._prefill_impl(ids[:, :30000], prefill_chunk_size=a.chunk,
                                do_score=False, window_size=a.window,
                                chunk_ratio=1.0, level="pair")
        finally:
            model.__dict__.pop("_init_kv", None)
        V = kv.value_cache[a.pool_layer][0]                # [Hkv,N,d]
        N = V.shape[1]
        flat = V.permute(1, 0, 2).reshape(N, -1).to(net.proj.weight.dtype)
        off = net.pool(flat)                    # 离线：`net.pool` 内部自带 proj
        onl = OnlineAttnPool(net.pool_q, device=V.device)
        for i in range(0, N, 7777):
            onl.update(net.proj(flat[i:i + 7777]))
        e = float((onl.value() - off).abs().max())
        del kv, V, flat
        torch.cuda.empty_cache()
    if not (e < 1e-3):
        fail(f"E 在线/离线池化在真 V 上不等价：max|Δ| = {e:.3e}")
    print(f"E ✓ 在线池化 ≡ 离线池化（真 V，N={N}，分 {(N+7776)//7777} 块）"
          f"max|Δ| = {e:.2e}")

    print("\n" + "=" * 72)
    print("ProMeta GPU 冒烟 A–E 全过。⚠ 这只证明**管道正确**，"
          "与方法有没有效果无关（Student 未训练）。")
    print("Finished.")


if __name__ == "__main__":
    main()
