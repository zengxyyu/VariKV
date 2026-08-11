"""P0-B：局部反事实损伤 × 全局行为分歧探针（NEXT_STEPS.md v5 §4）。

回答的问题：FastKVzip 驱逐掉的，是"低质量 + 低反差"的垃圾，还是"低质量但高反差"的信息？
以及：局部损伤经 W_O 投影后是否仍然大，且与最终行为分歧相关？

逐 (layer, kv_head, query_head, target_token) 记录：
    M = 1−λ = D_E/(D_R+D_E)        遗漏的 softmax 质量（用 logsumexp 算，不算 exp(lse)）
    C = ‖o_E − o_R‖                驱逐-保留反差
    G = ‖o_all − o_R‖ = M·C        局部损伤（value 空间）
    G_proj  = ‖W_O^(hq)·Δo‖        ← 跨 head/layer 唯一可比的量
    G_layer = ‖W_O·concat_hq(Δo)‖  ← 含跨头相消
query 位置用**数据集真实的问题 token**（不是上下文尾部——那些全在 local window 保护内，
驱逐伤不到，实测 B≈0.001，等于没有信号）。逐 query token 记录：
    B = KL(p_full ‖ p_pruned)      全局行为分歧（两次真实前向）

三条实现要求（v5 §4）：
  · logsumexp 算 M（169k 上下文下 exp(lse) 会溢出）
  · 逐 query_head，不能只逐 kv_head（GQA 7:1，平均会淹掉受影响的那个）
  · 用模型实际的 score：post-RoPE q/k、causal、以及保留集的**实际**定义 self.valid

恒等式自检：‖o_all−o_R‖ 必须等于 M·C（v5 §1.1 的恒等式只在同一轨迹、同一 q/K/V 下成立）。
不通过就 assert 失败——这是整份分析的地基。

挂钩方式：monkey-patch RetainCache.prepare 取 post-RoPE query。
RetainCache 在 prune_chunk 后把 flatten 翻成 True，因此 prepare 会被调用
（kvcache.py:326），参数正是 [B,HQ,T,d] 的 post-RoPE query。
进程内 patch，不影响正在跑的其它 job。
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip                    # noqa: E402
from attention.kvcache import RetainCache               # noqa: E402
from data.load import load_dataset_all                  # noqa: E402
from data.wrapper import DataWrapper, get_query          # noqa: E402

_QCAP = {}          # layer_idx -> post-RoPE query [B,HQ,T,d]
_orig_prepare = RetainCache.prepare


def _patched_prepare(self, query_states, key_states, value_states, layer_idx):
    _QCAP[layer_idx] = query_states.detach().clone()
    return _orig_prepare(self, query_states, key_states, value_states, layer_idx)


def get_valid(kv, layer_idx, S):
    """[H,S] 的 bool 保留掩码。兼容不同签名。"""
    try:
        v = kv._get_valid(layer_idx)
    except TypeError:
        v = kv._get_valid(layer_idx, S)
    if isinstance(v, (list, tuple)):
        v = torch.stack([x.bool() for x in v])
    v = v.bool()
    while v.dim() > 2:
        v = v.squeeze(0)
    if v.shape[-1] != S:                    # 尾部补 True（例如刚 update 进来的新 token）
        pad = torch.ones(v.shape[0], S - v.shape[-1], dtype=torch.bool, device=v.device)
        v = torch.cat([v, pad], dim=-1)
    return v


@torch.no_grad()
def layer_damage(kv, model, layer_idx, T, wo_weight):
    """返回该层的 per-(h_kv, g, t) 量，以及层级 G_layer [T]。"""
    q = _QCAP.get(layer_idx)
    if q is None:
        return None
    k_all = kv.key_cache[layer_idx]                      # [1,H,S,d]
    v_all = kv.value_cache[layer_idx]
    S = k_all.shape[2]
    if S <= T:
        return None
    H, d = k_all.shape[1], k_all.shape[3]
    B, HQ, Tq, _ = q.shape
    assert Tq == T, (Tq, T)
    Gq = HQ // H
    dev = k_all.device
    valid = get_valid(kv, layer_idx, S).to(dev)          # [H,S]

    # 目标 token 占据缓存最后 T 个位置 ⇒ query i 可见 key j ⇔ j ≤ S−T+i
    idx_k = torch.arange(S, device=dev).view(1, S)
    idx_q = (S - T) + torch.arange(T, device=dev).view(T, 1)
    causal = idx_k <= idx_q                              # [T,S]

    scale = 1.0 / (d ** 0.5)
    out = {"M": [], "C": [], "G": [], "Gproj": []}
    dy = torch.zeros(T, wo_weight.shape[0], device=dev, dtype=torch.float32)

    for h in range(H):
        qh = q[0].view(H, Gq, T, d)[h].float()           # [Gq,T,d]
        kh = k_all[0, h].float()                         # [S,d]
        vh = v_all[0, h].float()
        s = torch.einsum("gtd,sd->gts", qh, kh) * scale  # [Gq,T,S]
        neg = torch.finfo(torch.float32).min
        s = s.masked_fill(~causal.view(1, T, S), neg)
        vmask = valid[h].view(1, 1, S)
        s_R = s.masked_fill(~vmask, neg)
        s_E = s.masked_fill(vmask, neg)

        L_R = torch.logsumexp(s_R, dim=-1)               # [Gq,T]
        L_E = torch.logsumexp(s_E, dim=-1)
        L_F = torch.logaddexp(L_R, L_E)
        M = torch.exp(L_E - L_F)                         # [Gq,T]

        o_R = torch.einsum("gts,sd->gtd", torch.softmax(s_R, -1), vh)
        o_E = torch.einsum("gts,sd->gtd", torch.softmax(s_E, -1), vh)
        o_A = torch.einsum("gts,sd->gtd", torch.softmax(s, -1), vh)
        C = (o_E - o_R).norm(dim=-1)                     # [Gq,T]
        dlt = o_A - o_R                                  # [Gq,T,d]
        G = dlt.norm(dim=-1)

        # 恒等式自检：只在 M 不接近 0/1 且 o_E 有效（存在被驱逐 key）的位置查
        ok = (L_E > neg / 2) & (M > 1e-4) & (M < 1 - 1e-4)
        if ok.any():
            lhs, rhs = G[ok], (M * C)[ok]
            rel = ((lhs - rhs).abs() / rhs.clamp_min(1e-8)).max().item()
            assert rel < 2e-2, f"恒等式失败 layer{layer_idx} h{h}: 最大相对误差 {rel:.3e}"

        # W_O 投影：query head 编号 hq = h*Gq + g，取 o_proj 的对应列块
        for g in range(Gq):
            hq = h * Gq + g
            Wb = wo_weight[:, hq * d:(hq + 1) * d].float()      # [d_model, d]
            contrib = dlt[g] @ Wb.T                              # [T, d_model]
            out["Gproj"].append(contrib.norm(dim=-1).cpu())      # [T]
            dy += contrib
        out["M"].append(M.cpu()); out["C"].append(C.cpu()); out["G"].append(G.cpu())
        del s, s_R, s_E, o_R, o_E, o_A, dlt
        torch.cuda.empty_cache()

    return {
        "M": torch.cat(out["M"]),            # [H*Gq, T]
        "C": torch.cat(out["C"]),
        "G": torch.cat(out["G"]),
        "Gproj": torch.stack(out["Gproj"]),  # [H*Gq, T]
        "Glayer": dy.norm(dim=-1).cpu(),     # [T]  含跨头相消
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--gate", default="fastkvzip")
    ap.add_argument("--data", default="scbench_kv")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--chunk", type=int, default=16000)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--level", default="pair")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--n_queries", type=int, default=5,
                    help="每样本用几个真实问题作为 query 位置")
    ap.add_argument("--out", default="scratch_probe_damage.pt")
    args = ap.parse_args()

    RetainCache.prepare = _patched_prepare
    m = ModelKVzip(args.model, kv_type="retain", gate_path_or_name=args.gate)
    L = m.config.num_hidden_layers
    wo = [m.model.model.layers[l].self_attn.o_proj.weight for l in range(L)]
    print(f"[cfg] L={L} H_kv={m.config.num_key_value_heads} "
          f"HQ={m.config.num_attention_heads} ratio={args.ratio}",
          flush=True)

    ds = load_dataset_all(args.data, m.tokenizer)
    dw = DataWrapper(args.data, ds, m)
    recs = []

    for si in range(args.n_samples):
        # ---- 上下文只前填一次（两种 cache 分开跑，避免同时驻留 2×9.7GB）
        questions = list(ds[si]["question"])[: args.n_queries]
        q_ids_list = [m.apply_template(get_query("qa", q)) for q in questions]

        # (1) 满缓存：拿各问题的 logits
        kv_f = dw.prefill_context(si, do_score=False)
        S_f = kv_f.key_cache[0].shape[2]
        lg_full = []
        for qi in q_ids_list:
            lg_full.append(m.model(qi.to(m.device),
                                   past_key_values=kv_f).logits[0].float().cpu())
            kv_f.slice(S_f)
        n_ctx = S_f
        del kv_f; torch.cuda.empty_cache()

        # (2) 分块前填 + 驱逐：拿 logits 与逐层局部损伤
        kv_p = dw.prefill_context(si, prefill_chunk=args.chunk,
                                  window_size=args.window,
                                  chunk_ratio=args.ratio, level=args.level)
        S_p = kv_p.key_cache[0].shape[2]
        per_q = []
        for qi, (qids, lgf) in enumerate(zip(q_ids_list, lg_full)):
            _QCAP.clear()
            T = qids.shape[-1]
            lgp = m.model(qids.to(m.device), past_key_values=kv_p).logits[0].float()
            B = torch.nn.functional.kl_div(
                torch.log_softmax(lgp, -1), torch.log_softmax(lgf.to(m.device), -1),
                reduction="none", log_target=True).sum(-1).cpu()          # [T]
            layers = {}
            for l in range(L):
                r = layer_damage(kv_p, m, l, T, wo[l])
                if r is not None:
                    layers[l] = r
            per_q.append({"q": qi, "T": int(T), "B": B, "layers": layers})
            print(f"  样本{si} 问题{qi}: T={T}  B[last]={B[-1]:.4f}  "
                  f"B[max]={B.max():.4f}  记录层 {len(layers)}", flush=True)
            kv_p.slice(S_p)
            del lgp; torch.cuda.empty_cache()
        recs.append({"sample": si, "n_ctx": int(n_ctx), "queries": per_q})
        torch.save(recs, args.out)
        print(f"样本 {si} 完成，ctx {n_ctx} tok，已存 {args.out}"
              f"（{len(recs)} 样本）", flush=True)
        del kv_p; torch.cuda.empty_cache()

    print("\nDONE 恒等式自检全部通过。", flush=True)


if __name__ == "__main__":
    main()
