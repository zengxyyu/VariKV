"""定位 tier5 (absorb_mode=dist) 在 nd=2000（34k token）上的 NaN。

现象：tier5 在 nd=0/200/800 正常，只有 nd=2000 出 nan；tier2/4（point 吸收）
在同样长度上正常。所以嫌疑集中在 dist 特有的方差/精度递推，且要累积够多
chunk（34357/512 ≈ 67 块）才发作。

方法：走**真实**的 mem.prefill()（不手搓循环，免得引入自己的 bug），
用逐渐加长的上下文截断做二分，找出第一个出问题的长度，再看是哪个状态量先坏。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from varikv.config import Config
from varikv.train import build, encode_sample, forward_loss
from stage1 import data as stage1_data

TIER = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LEVEL = 2000

cfg = Config()
cfg.cache.budget = 256
cfg = cfg.ablation(TIER)
model, tok, mem = build(cfg)
mem.eval()
ck = Path(f"varikv/ckpt/k16_tier{TIER}.pt")
if ck.exists():
    mem.load_state_dict(torch.load(ck, map_location=cfg.device)["memory"])
    print(f"loaded {ck}")

s = [x for x in stage1_data.load("stage1/val.jsonl") if x.n_distract == LEVEL][0]
ctx, q, a = encode_sample(tok, s, cfg.device)
T = ctx.shape[1]
print(f"tier={TIER} evict={cfg.cache.evict_policy} absorb={cfg.cache.absorb_mode}")
print(f"ctx={T} budget={cfg.cache.budget} chunk={cfg.cache.prefill_chunk} "
      f"→ {T // cfg.cache.prefill_chunk} 块\n")

M = mem.memory
NAMES = ("mu", "logvar", "tau", "pos", "s2", "sum_tau_mu")


def probe(prefix):
    bad_any = False
    parts = []
    for name in NAMES:
        t = getattr(M, name, None)
        if not torch.is_tensor(t):
            continue
        f = t.float()
        nan, inf = int(torch.isnan(f).sum()), int(torch.isinf(f).sum())
        if nan or inf:
            bad_any = True
            parts.append(f"{name}:NaN={nan},Inf={inf}")
        else:
            parts.append(f"{name}:[{f.min():.2e},{f.max():.2e}]")
    print(f"  {prefix}  " + "  ".join(parts))
    return bad_any


lengths = [2048, 4096, 8192, 16384, 24576, T]
with torch.no_grad():
    for L in lengths:
        L = min(L, T)
        sub = ctx[:, :L]
        mem.prefill(model, sub)
        bad = probe(f"L={L:6d} n_mem={mem.n_mem:3d}")
        nll = forward_loss(model, mem, sub, q, a).item()
        print(f"           nll={nll:.4f}   {'<<< 坏了' if (bad or nll != nll) else ''}")
        if bad or nll != nll:
            print(f"\n*** 第一次出问题的长度：L={L} ***")
            break
