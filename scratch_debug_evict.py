"""诊断：自由能驱逐为什么不如朴素的 recency，且 K 越大越糟。

已知（2026-08-07 扫描，只算被压缩的样本）：
    K=16  t4(fe+point) 2.8114  vs  t2(recency+point) 2.8612   → fe 好 0.05
    K=32  t4 3.0601            vs  t2 2.9252                  → fe 差 0.14
    K=64  t4 3.6710            vs  t2 3.2997                  → fe 差 0.37

三个待查假设：
  H1 摊销预测器的排序质量差 → 驱逐等于随机
  H2 fe 把「针」（答案所在的 KV）驱逐掉了，而 recency 反而保住了
  H3 fe 的打分退化成了某种和位置强相关的东西（即白白复杂化）
"""
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from varikv.config import Config
from varikv.train import build, encode_sample
from stage1 import data as stage1_data

K = int(sys.argv[1]) if len(sys.argv) > 1 else 16
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CKDIR = sys.argv[3] if len(sys.argv) > 3 else "varikv/ckpt"
LEVEL = 800


def spearman(a, b):
    """秩相关，避免引入 scipy。"""
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db + 1e-12)


def needle_token_ids(tok, sample):
    """答案所在的 token 位置。update 型取**最后**一次赋值（那才是正确答案）。"""
    text = sample.context
    ch = text.rfind(sample.answer)
    if ch < 0:
        return set()
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    hit = set()
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if e > ch and s < ch + len(sample.answer):
            hit.add(i)
    return hit


val = [s for s in stage1_data.load("stage1/val.jsonl") if s.n_distract == LEVEL][:N]
print(f"K={K}  level={LEVEL}  n={len(val)}\n")

for tier, label in ((2, "recency"), (4, "free_energy")):
    cfg = Config()
    cfg.cache.budget = 256
    cfg.memory.num_slots = K
    cfg = cfg.ablation(tier)
    model, tok, mem = build(cfg)
    mem.eval()
    ck = Path(CKDIR) / f"k{K}_tier{tier}.pt"
    if not ck.exists(): ck = Path("varikv/ckpt") / f"k{K}_tier{tier}.pt"
    if ck.exists():
        mem.load_state_dict(torch.load(ck, map_location=cfg.device)["memory"])

    keep_rate, pos_frac, needle_keep = [], [], []
    with torch.no_grad():
        for s in val:
            ctx, q, a = encode_sample(tok, s, cfg.device)
            nd = needle_token_ids(tok, s)
            mem.prefill(model, ctx)
            tp = mem.token_pos
            if tp is None:
                print("  token_pos 为空，无法追踪")
                break
            kept = set(int(x) for x in tp.flatten().tolist())
            T = ctx.shape[1]
            keep_rate.append(len(kept) / T)
            # 保留下来的位置在上下文里的相对位置（1.0=全在末尾）
            pos_frac.append(sum(kept) / (len(kept) * max(T - 1, 1)))
            if nd:
                needle_keep.append(len(nd & kept) / len(nd))

    n = len(keep_rate)
    print(f"=== tier {tier} ({label}) ===")
    print(f"  保留比例          {sum(keep_rate)/n:.4f}")
    print(f"  保留位置的平均相对位置  {sum(pos_frac)/n:.4f}   (1.0 = 全部集中在末尾)")
    if needle_keep:
        nk = sum(needle_keep) / len(needle_keep)
        print(f"  **针的保留率**     {nk:.4f}   ({sum(1 for x in needle_keep if x>0)}/{len(needle_keep)} 个样本至少保住一部分)")
    print()
    del model, mem
    torch.cuda.empty_cache()
