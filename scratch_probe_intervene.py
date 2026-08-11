"""P0-B'：top-head 因果 intervention（NEXT_STEPS.md v5 §4）。

探针给的是**相关**：G_proj 大的 head 是否真的造成了行为分歧？这个脚本给**因果**。

做法：在压缩轨迹上，把选定 head 的注意力输出从 o_R 改回 o_all（即"该 head 若能看到自己
被驱逐的 KV 会输出什么"），再看最终 KL(p_full ‖ p_intervened) 相对未干预时降了多少。

实现要点：在 `o_proj` 上挂 **forward pre-hook**。它的输入是各 head 输出的拼接
[B,T,HQ*d]，往选定 head 的那一段 slice 上加 Δo_h 即可——经过 o_proj 之后正好等于
把 W_O^(h)Δo_h 加进残差流，与 v5 §1.6 的定义严格一致，不需要改任何共享代码。

判读（v5 的 Q1）：
  · G_proj 大但 restore 后 B 几乎不变 ⇒ 局部损伤不是任务关键 ⇒ 整个后驱逐重建方向危险
  · G_proj 中等但 restore 后 B 明显下降 ⇒ 真正重要，值得继续
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip                    # noqa: E402
from attention.kvcache import RetainCache               # noqa: E402
from data.load import load_dataset_all                  # noqa: E402
from data.wrapper import DataWrapper, get_query          # noqa: E402

_QCAP = {}
_orig_prepare = RetainCache.prepare
_STATE = {"kv": None, "heads": defaultdict(list), "on": False}


def _patched_prepare(self, query_states, key_states, value_states, layer_idx):
    _QCAP[layer_idx] = query_states.detach().clone()
    return _orig_prepare(self, query_states, key_states, value_states, layer_idx)


def get_valid(kv, layer_idx, S):
    try:
        v = kv._get_valid(layer_idx)
    except TypeError:
        v = kv._get_valid(layer_idx, S)
    if isinstance(v, (list, tuple)):
        v = torch.stack([x.bool() for x in v])
    v = v.bool()
    while v.dim() > 2:
        v = v.squeeze(0)
    if v.shape[-1] != S:
        pad = torch.ones(v.shape[0], S - v.shape[-1], dtype=torch.bool, device=v.device)
        v = torch.cat([v, pad], dim=-1)
    return v


@torch.no_grad()
def delta_o(kv, layer_idx, T, hq_list):
    """返回 {hq: Δo [T,d]}，Δo = o_all − o_R，在同一 q/K/V 上算（同状态反事实）。"""
    q = _QCAP.get(layer_idx)
    if q is None:
        return {}
    k_all, v_all = kv.key_cache[layer_idx], kv.value_cache[layer_idx]
    S, H, d = k_all.shape[2], k_all.shape[1], k_all.shape[3]
    HQ = q.shape[1]; Gq = HQ // H
    dev = k_all.device
    valid = get_valid(kv, layer_idx, S).to(dev)
    idx_k = torch.arange(S, device=dev).view(1, S)
    idx_q = (S - T) + torch.arange(T, device=dev).view(T, 1)
    causal = idx_k <= idx_q
    neg = torch.finfo(torch.float32).min
    scale = 1.0 / (d ** 0.5)
    out = {}
    for hq in hq_list:
        h, g = hq // Gq, hq % Gq
        qh = q[0].view(H, Gq, T, d)[h, g].float()            # [T,d]
        kh = k_all[0, h].float(); vh = v_all[0, h].float()
        s = (qh @ kh.T) * scale                              # [T,S]
        s = s.masked_fill(~causal, neg)
        vm = valid[h].view(1, S)
        o_R = torch.softmax(s.masked_fill(~vm, neg), -1) @ vh
        o_A = torch.softmax(s, -1) @ vh
        out[hq] = (o_A - o_R)
        del s, o_R, o_A
    torch.cuda.empty_cache()
    return out


def make_hook(layer_idx):
    def pre_hook(module, args):
        if not _STATE["on"]:
            return None
        hq_list = _STATE["heads"].get(layer_idx)
        if not hq_list:
            return None
        x = args[0]                                          # [B,T,HQ*d]
        T = x.shape[1]
        d = _STATE["d"]
        deltas = delta_o(_STATE["kv"], layer_idx, T, hq_list)
        x = x.clone()
        for hq, dl in deltas.items():
            x[0, :, hq * d:(hq + 1) * d] += dl.to(x.dtype)
        return (x,) + tuple(args[1:])
    return pre_hook


def pick_heads(path, topk, mode):
    """从探针结果里挑 head。mode: top / low"""
    recs = torch.load(path, map_location="cpu", weights_only=False)
    acc = defaultdict(list)
    for r in recs:
        for q in r["queries"]:
            for l, dd in q["layers"].items():
                gp = dd["Gproj"].numpy()                      # [HQ,T]
                for hq in range(gp.shape[0]):
                    acc[(l, hq)].append(float(gp[hq].mean()))
    means = {k: float(np.mean(v)) for k, v in acc.items()}
    order = sorted(means, key=lambda k: -means[k])
    sel = order[:topk] if mode == "top" else order[-topk:]
    return sel, means


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
    ap.add_argument("--probe", default="scratch_probe_damage.pt")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--curve", type=int, nargs="*", default=None,
                    help="恢复曲线：给出若干 top-N（如 5 20 80），并自动加 all 与 low 对照。"
                         "all 量的是'局部反事实修正能解释多少行为分歧'——即局部可修 vs 继承漂移")
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--n_queries", type=int, default=3)
    args = ap.parse_args()

    top_heads, means = pick_heads(args.probe, args.topk, "top")
    low_heads, _ = pick_heads(args.probe, args.topk, "low")
    print(f"[选头] top-{args.topk} by G_proj: "
          + ", ".join(f"L{l}H{h}({means[(l,h)]:.4f})" for l, h in top_heads))
    print(f"[选头] 低 G_proj 对照:            "
          + ", ".join(f"L{l}H{h}({means[(l,h)]:.4f})" for l, h in low_heads), flush=True)

    RetainCache.prepare = _patched_prepare
    m = ModelKVzip(args.model, kv_type="retain", gate_path_or_name=args.gate)
    L = m.config.num_hidden_layers
    _STATE["d"] = getattr(m.config, "head_dim",
                          m.config.hidden_size // m.config.num_attention_heads)
    for l in range(L):
        m.model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_hook(l))

    ds = load_dataset_all(args.data, m.tokenizer)
    dw = DataWrapper(args.data, ds, m)

    if args.curve:
        arms = {"none": []}
        for n in args.curve:
            arms[f"top{n}"] = pick_heads(args.probe, n, "top")[0]
        arms["all"] = [(l, h) for l in range(L) for h in range(m.config.num_attention_heads)]
        arms["low"] = low_heads
    else:
        arms = {"none": [], "top": top_heads, "low": low_heads}
    res = {k: [] for k in arms}

    for si in range(args.n_samples):
        questions = list(ds[si]["question"])[: args.n_queries]
        q_ids_list = [m.apply_template(get_query("qa", q)) for q in questions]

        kv_f = dw.prefill_context(si, do_score=False)
        S_f = kv_f.key_cache[0].shape[2]
        lg_full = []
        for qi in q_ids_list:
            lg_full.append(m.model(qi.to(m.device),
                                   past_key_values=kv_f).logits[0].float().cpu())
            kv_f.slice(S_f)
        del kv_f; torch.cuda.empty_cache()

        kv_p = dw.prefill_context(si, prefill_chunk=args.chunk,
                                  window_size=args.window,
                                  chunk_ratio=args.ratio, level=args.level)
        S_p = kv_p.key_cache[0].shape[2]
        _STATE["kv"] = kv_p

        for qi, (qids, lgf) in enumerate(zip(q_ids_list, lg_full)):
            lgf_d = lgf.to(m.device)
            for arm, heads in arms.items():
                _STATE["heads"].clear()
                for l, hq in heads:
                    _STATE["heads"][l].append(hq)
                _STATE["on"] = bool(heads)
                _QCAP.clear()
                lgp = m.model(qids.to(m.device), past_key_values=kv_p).logits[0].float()
                B = torch.nn.functional.kl_div(
                    torch.log_softmax(lgp, -1), torch.log_softmax(lgf_d, -1),
                    reduction="none", log_target=True).sum(-1)
                res[arm].append((float(B[-1]), float(B.mean()), float(B.max())))
                _STATE["on"] = False
                kv_p.slice(S_p)
                del lgp; torch.cuda.empty_cache()
            print(f"  样本{si} 问题{qi}: "
                  + "  ".join(f"{a} B[last]={res[a][-1][0]:.4f}" for a in arms),
                  flush=True)
        del kv_p; torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print(f"{'arm':>6} {'n':>4} {'B[last]':>10} {'B[mean]':>10} {'B[max]':>10} {'vs none':>12}")
    base = np.array([r[0] for r in res["none"]])
    for a in arms:
        v = np.array(res[a])
        rel = (v[:, 0] - base).mean() / max(base.mean(), 1e-12) * 100
        print(f"{a:>6} {len(v):>4} {v[:,0].mean():>10.4f} {v[:,1].mean():>10.4f} "
              f"{v[:,2].mean():>10.4f} {rel:>11.1f}%")
    print("=" * 78)
    print("判读：top 相对 none 的 B 显著下降 ⇒ 局部损伤是因果的、值得恢复；")
    print("      几乎不变 ⇒ 局部损伤不是任务关键，整个后驱逐重建方向危险（v5 Q1）。")


if __name__ == "__main__":
    main()
