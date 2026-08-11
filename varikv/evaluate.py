"""五档消融评测（§11.7）。

    tier 1  discard                 丢弃              KVzip / FastKVzip
    tier 2  recency     + point     新近+点吸收        Infini-attention / Tensor Cache
    tier 3  recency     + moment    新近+二阶矩        MomentKV（training-free）← 真实门槛
    tier 4  free_energy + point     自由能+点吸收      IndexMem 加强
    tier 5  free_energy + dist      自由能+分布吸收    VariKV

关键对比：
    2→3  二阶矩相对点均值的增益（说明「存方差」这件事本身值多少）
    3→5  **生死问题**：贝叶斯信念 vs 频率派矩统计，二阶信息两边都有，
         自变量收敛到「KL 门控 + 方差感知读出」是否真的有用
    4→5  自由能驱逐固定时，分布式吸收相对点吸收的增益
结果按 kind(retain/update) × n_distract 分组：理论预测干扰强度越高、
分布式相对点记忆的优势越大（stage1/data.py 的设计动机）。
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from varikv.config import Config
from varikv.train import build, encode_sample, forward_loss
from stage1 import data as stage1_data


@torch.no_grad()
def generate(model, mem, tok, ctx_ids, q_ids, max_new_tokens=48):
    # 16 太短：模型会先说一句 "The current value of user_X is ..." 再给答案，
    # 开场白就吃掉十几个 token，答案永远来不及出现。实测 16→48 使 tier1 在
    # 不触发驱逐的样本上从 0.00 升到 0.96。
    cache = mem.prefill(model, ctx_ids)
    cur = q_ids
    out_ids = []
    for _ in range(max_new_tokens):
        B, T = cur.shape
        kv_len = mem.n_mem + mem.n_seen
        position_ids = torch.arange(
            mem.n_seen, mem.n_seen + T, device=cur.device
        ).unsqueeze(0).expand(B, T)
        attn = torch.ones(B, kv_len + T, device=cur.device, dtype=torch.long)
        out = model(
            input_ids=cur, past_key_values=cache, position_ids=position_ids,
            attention_mask=attn, use_cache=True,
        )
        cache = out.past_key_values
        mem.n_seen += T
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        if nxt.item() == tok.eos_token_id:
            break
        out_ids.append(nxt.item())
        cur = nxt
    return tok.decode(out_ids, skip_special_tokens=True)


def exact_match(pred: str, gold: str) -> bool:
    # 值的词表是 形容词-动物-两位数，答案可精确匹配（stage1/data.py）
    return gold.strip().lower() in pred.strip().lower()


@torch.no_grad()
def evaluate(cfg: Config, tier: int, ckpt: str = None, limit: int = None,
             per_level: int = 0, levels=None, do_gen: bool = True):
    """主指标是 nll（answer token 上的交叉熵），em 只作辅助。

    为什么不能用 em 当主指标（2026-08-07 实测）：把 3.5k 上下文压进 K 个 latent
    高斯槽是 ~377:1，而答案是「形容词-动物-两位数」这样的高熵随机串，逐字符复原
    对**任何**有损记忆都不可能 —— 实测五档 em 全 0.000，指标没有分辨率。
    同一批样本上 nll 却把五档清楚分开（无记忆 5.04 → 有记忆 2.60），且生成文本
    显示记忆确实带回了部分信息（gold `jade-shrike-85` → 生成 `jade-otter-59`，
    形容词对了）。所以 nll 才是能回答 GO/NO-GO 的读数。
    """
    cfg = cfg.ablation(tier)
    model, tok, mem = build(cfg)
    mem.eval()

    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location=cfg.device)
        mem.load_state_dict(state["memory"])
        print(f"loaded {ckpt}")
    elif tier not in (1, 3):
        # tier 1 丢弃、tier 3 training-free，本就没有 checkpoint
        print(f"[warn] tier {tier} 没有加载 checkpoint，评的是随机初始化的记忆模块")

    val = stage1_data.load(str(Path(__file__).parent.parent / "stage1/val.jsonl"))
    if levels:
        val = [s for s in val if s.n_distract in levels]
    if per_level:
        # 每个干扰档取同样多的样本，避免各档样本量不均导致 overall 被某档主导
        by_level = defaultdict(list)
        for s in val:
            by_level[s.n_distract].append(s)
        val = [s for lv in sorted(by_level) for s in by_level[lv][:per_level]]
    elif limit:
        val = val[:limit]

    # [n_em_correct, n, sum_nll]
    buckets = defaultdict(lambda: [0, 0, 0.0])
    # 逐样本记录：各档看的是同一批样本，故可做**配对**比较。tier5 相对 tier2/4
    # 的差距只有 ~0.02 nats，非配对的均值对比分辨不出它是不是噪声。
    per_sample = []
    for i, s in enumerate(val):
        ctx, q, a = encode_sample(tok, s, cfg.device)
        nll = forward_loss(model, mem, ctx, q, a).item()
        ok = int(exact_match(generate(model, mem, tok, ctx, q), s.answer)) if do_gen else 0
        per_sample.append({"i": i, "kind": s.kind, "level": s.n_distract,
                           "nll": nll, "em": ok})
        for key in [(s.kind, s.n_distract), (s.kind, "all"), ("all", s.n_distract)]:
            buckets[key][0] += ok
            buckets[key][1] += 1
            buckets[key][2] += nll

    results = {f"{k[0]}/{k[1]}": {"em": c / max(n, 1), "nll": t / max(n, 1), "n": n}
               for k, (c, n, t) in buckets.items()}
    results["_per_sample"] = per_sample
    tot_n = sum(v[1] for k, v in buckets.items() if k[0] == "all")
    results["overall"] = {
        "em": sum(v[0] for k, v in buckets.items() if k[0] == "all") / max(tot_n, 1),
        "nll": sum(v[2] for k, v in buckets.items() if k[0] == "all") / max(tot_n, 1),
        "n": tot_n,
    }
    return results


def ckpt_path(ckpt_dir: str, tier: int, num_slots: int) -> Path:
    """K=16 保留旧文件名以兼容早先的 ckpt，其余带 k{K}_ 前缀。"""
    d = Path(ckpt_dir)
    pref = d / f"k{num_slots}_tier{tier}.pt"
    if pref.exists():
        return pref
    legacy = d / f"tier{tier}.pt"
    if num_slots == 16 and legacy.exists():
        return legacy
    return pref


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--num_slots", type=int, default=None, help="K：每个 (layer,kv_head) 的槽数")
    ap.add_argument("--ckpt_dir", type=str, default="varikv/ckpt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per_level", type=int, default=0, help="每个干扰档取多少样本（均衡采样）")
    ap.add_argument("--levels", type=int, nargs="+", default=None)
    ap.add_argument("--no_gen", action="store_true", help="只算 nll，跳过生成（省时间）")
    ap.add_argument("--json_out", type=str, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.model:
        cfg.model_name = args.model
    if args.budget:
        cfg.cache.budget = args.budget
    if args.num_slots:
        cfg.memory.num_slots = args.num_slots
    K = cfg.memory.num_slots

    all_res = {}
    for t in args.tier:
        ck = ckpt_path(args.ckpt_dir, t, K)
        res = evaluate(cfg, t, str(ck), args.limit, args.per_level, args.levels,
                       do_gen=not args.no_gen)
        all_res[t] = res
        print(f"\n=== tier {t}  (K={K}) ===")
        for k in sorted(res):
            if k.startswith("_"):
                continue
            v = res[k]
            print(f"  {k:24s} nll {v['nll']:7.4f}   em {v['em']:.3f}   n={v['n']}")

    print(f"\n=== 五档对比  K={K}  (主指标 nll，越低越好) ===")
    base = all_res.get(1, {}).get("overall", {}).get("nll")
    for t in sorted(all_res):
        o = all_res[t]["overall"]
        d = f"   Δ vs tier1 {o['nll'] - base:+.4f}" if base is not None else ""
        print(f"  tier {t}: nll {o['nll']:7.4f}   em {o['em']:.3f}{d}")

    payload = {"num_slots": K, "budget": cfg.cache.budget, "tiers": all_res}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=1))
    # stdout 只打摘要，逐样本记录太长，留在 --json_out 里
    print(json.dumps({t: {k: v for k, v in r.items() if not k.startswith("_")}
                      for t, r in all_res.items()}, indent=1))
