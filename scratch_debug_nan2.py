"""复现并定位 tier5 在 val 样本 i=144（nd=2000）上的 NaN。

160 个里只坏 1 个，且三个 K 都是同一个 → 确定性地由该输入触发，不是随机数值噪声。
tier2/4（point 吸收）在同一样本上正常 → dist 特有。
"""
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from varikv.config import Config
from varikv.train import build, encode_sample, forward_loss
from stage1 import data as stage1_data

TIER = int(sys.argv[1]) if len(sys.argv) > 1 else 5
IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 144
PER_LEVEL = 40

cfg = Config()
cfg.cache.budget = 256
cfg = cfg.ablation(TIER)
model, tok, mem = build(cfg)
mem.eval()
ck = Path(f"varikv/ckpt/k16_tier{TIER}.pt")
if ck.exists():
    mem.load_state_dict(torch.load(ck, map_location=cfg.device)["memory"])

# 复刻 evaluate.py 的均衡采样，取到同一个样本
val = stage1_data.load("stage1/val.jsonl")
by = defaultdict(list)
for s in val:
    by[s.n_distract].append(s)
val = [s for lv in sorted(by) for s in by[lv][:PER_LEVEL]]
s = val[IDX]
print(f"tier={TIER} idx={IDX} kind={s.kind} nd={s.n_distract} answer={s.answer!r}")

ctx, q, a = encode_sample(tok, s, cfg.device)
T = ctx.shape[1]
print(f"ctx={T}  answer_tokens={a.shape[1]}  q_tokens={q.shape[1]}\n")

M = mem.memory


def probe(tag):
    parts, bad = [], False
    for name in ("mu", "logvar", "tau", "pos", "s2"):
        t = getattr(M, name, None)
        if not torch.is_tensor(t):
            continue
        f = t.float()
        n, i = int(torch.isnan(f).sum()), int(torch.isinf(f).sum())
        if n or i:
            bad = True
            parts.append(f"{name}:NaN={n},Inf={i}")
        else:
            parts.append(f"{name}:[{f.min():.2e},{f.max():.2e}]")
    print(f"  {tag}  " + "  ".join(parts))
    return bad


with torch.no_grad():
    for L in [4096, 8192, 16384, 24576, 30720, T]:
        L = min(L, T)
        mem.prefill(model, ctx[:, :L])
        bad_mem = probe(f"L={L:6d} n_mem={mem.n_mem:3d}")
        nll = forward_loss(model, mem, ctx[:, :L], q, a).item()
        flag = "  <<< 坏了" if (bad_mem or nll != nll) else ""
        print(f"           nll={nll:.4f}{flag}")
        if bad_mem or nll != nll:
            print(f"\n*** L={L} 开始出问题；记忆状态坏? {bad_mem} ***")
            if not bad_mem:
                print("    记忆本身干净 → NaN 出在读出的等效 KV 或后续前向，"
                      "不在贝叶斯递推里")
                eff = M.read().float()
                print(f"    read() 等效KV: NaN={int(torch.isnan(eff).sum())} "
                      f"Inf={int(torch.isinf(eff).sum())} "
                      f"range=[{eff.min():.3e},{eff.max():.3e}]")
                prec = M.read_precision().float()
                print(f"    read_precision: NaN={int(torch.isnan(prec).sum())} "
                      f"range=[{prec.min():.3e},{prec.max():.3e}]")
            break
        if L >= T:
            print("\n未复现 —— 该样本在此路径下正常")
