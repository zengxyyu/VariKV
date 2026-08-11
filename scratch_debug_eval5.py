"""区分两种可能：记忆链路在评测时是死的，还是活着但任务太难。

exact-match 太钝：把 3.5k 上下文压进 16 个 64 维高斯槽（~377:1），再要求
逐字符复原 "crimson-kite-33" 这种随机串，任何有损记忆都做不到。
所以改用连续指标——答案 token 上的 lm_loss——它对「记忆有没有带来信息」敏感得多。
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from varikv.config import Config
from varikv.train import build, encode_sample, forward_loss
from varikv.evaluate import generate
from stage1 import data as stage1_data

LEVEL = 200
N = 12
TIERS = [1, 2, 3, 4, 5]

base = Config()
base.cache.budget = 256
val = [s for s in stage1_data.load(str(Path(__file__).parent / "stage1/val.jsonl"))
       if s.n_distract == LEVEL][:N]

print(f"n_distract={LEVEL}  n={len(val)}  budget={base.cache.budget}\n")
summary = {}
for tier in TIERS:
    cfg = Config()
    cfg.cache.budget = 256
    cfg = cfg.ablation(tier)
    model, tok, mem = build(cfg)
    mem.eval()
    ck = Path(f"varikv/ckpt/tier{tier}.pt")
    if ck.exists():
        mem.load_state_dict(torch.load(ck, map_location=cfg.device)["memory"])

    losses, nmems, texts = [], [], []
    with torch.no_grad():
        for s in val:
            ctx, q, a = encode_sample(tok, s, cfg.device)
            l = forward_loss(model, mem, ctx, q, a)
            losses.append(l.item())
            nmems.append(mem.n_mem)
            if len(texts) < 2:
                texts.append(generate(model, mem, tok, ctx, q))
    summary[tier] = (sum(losses) / len(losses), nmems[0])
    print(f"=== tier {tier}  evict={cfg.cache.evict_policy} absorb={cfg.cache.absorb_mode} ===")
    print(f"  n_mem（读回的等效 KV 个数）= {nmems[0]}   n_seen={mem.n_seen}  ctx={ctx.shape[1]}")
    print(f"  answer 上的 lm_loss  mean={summary[tier][0]:.4f}")
    print(f"  gold={val[0].answer!r}")
    for t in texts:
        print(f"  gen : {t[:110]!r}")
    print()
    del model, mem
    torch.cuda.empty_cache()

print("=" * 62)
print(f"{'tier':>5} {'n_mem':>7} {'lm_loss':>10}   相对 tier1")
base_l = summary[1][0]
for t in TIERS:
    l, nm = summary[t]
    print(f"{t:>5} {nm:>7} {l:>10.4f}   {l - base_l:+.4f}")
