"""四档消融评测（§11.7）。

    tier 1  discard                丢弃            KVzip / FastKVzip
    tier 2  recency     + point    新近+点吸收      Infini-attention / Tensor Cache
    tier 3  free_energy + point    自由能+点吸收    IndexMem 加强
    tier 4  free_energy + dist     自由能+分布吸收  本方法

逐档递增 → 「自由能驱逐」与「分布式吸收」两个组件各自的增益被隔离证明。
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
from varikv.train import build, encode_sample
from stage1 import data as stage1_data


@torch.no_grad()
def generate(model, mem, tok, ctx_ids, q_ids, max_new_tokens=16):
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
def evaluate(cfg: Config, tier: int, ckpt: str = None, limit: int = None):
    cfg = cfg.ablation(tier)
    model, tok, mem = build(cfg)
    mem.eval()

    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location=cfg.device)
        mem.load_state_dict(state["memory"])
        print(f"loaded {ckpt}")
    elif tier != 1:
        print(f"[warn] tier {tier} 没有加载 checkpoint，评的是随机初始化的记忆模块")

    val = stage1_data.load(str(Path(__file__).parent.parent / "stage1/val.jsonl"))
    if limit:
        val = val[:limit]

    buckets = defaultdict(lambda: [0, 0])
    for s in val:
        ctx, q, _ = encode_sample(tok, s, cfg.device)
        pred = generate(model, mem, tok, ctx, q)
        ok = exact_match(pred, s.answer)
        for key in [(s.kind, s.n_distract), (s.kind, "all"), ("all", s.n_distract)]:
            buckets[key][0] += int(ok)
            buckets[key][1] += 1

    results = {f"{k[0]}/{k[1]}": c / max(n, 1) for k, (c, n) in buckets.items()}
    results["overall"] = sum(v[0] for k, v in buckets.items() if k[0] == "all" ) / max(
        sum(v[1] for k, v in buckets.items() if k[0] == "all"), 1
    )
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--ckpt_dir", type=str, default="varikv/ckpt")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.model:
        cfg.model_name = args.model
    if args.budget:
        cfg.cache.budget = args.budget

    all_res = {}
    for t in args.tier:
        ck = Path(args.ckpt_dir) / f"tier{t}.pt"
        res = evaluate(cfg, t, str(ck), args.limit)
        all_res[t] = res
        print(f"\n=== tier {t} ===")
        for k in sorted(res):
            print(f"  {k:24s} {res[k]:.3f}")

    print("\n=== 四档对比（overall）===")
    for t in sorted(all_res):
        print(f"  tier {t}: {all_res[t]['overall']:.3f}")
    print(json.dumps(all_res, indent=2))
