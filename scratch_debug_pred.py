"""决定性测量：摊销预测器的排序质量随 K 怎么变？

关键事实（free_energy.py:239）：**评测时 score() 只跑预测器**，精确 F 根本不算。
所以驱逐质量 == 预测器的排序质量。而 rel_pos（相对位置）是预测器的直接输入之一，
一旦它预测不准 F，最省力的退路就是照着位置排 —— 那就退化成 recency。

另一个可疑处：预测器看到的记忆状态是 memory_summary()，即
    cat([mu.mean(over K slots), logvar.mean(over K slots)])
维度恒为 2·d_z，**与 K 无关**。K 越大，这个"对 K 个槽求平均"的摘要越糊，
而它要预测的量（对 K 分量混合的 KL）却越复杂。两头背离。

量三件事：
  ρ(pred, exact)  预测器还原精确 F 排序的能力 —— 这是驱逐质量的上限
  ρ(pred, pos)    预测器有多依赖位置 —— 越高越像 recency
  ρ(exact, pos)   精确 F 本身有多像 recency —— 作为对照基准
"""
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from varikv.config import Config
from varikv.train import build, encode_sample
from stage1 import data as stage1_data

LEVEL = 800
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for p, i in enumerate(order):
            r[i] = float(p)
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db + 1e-12)


val = [s for s in stage1_data.load("stage1/val.jsonl") if s.n_distract == LEVEL][:N]
print(f"level={LEVEL}  n={N}   tier4 (free_energy + point)，评测模式\n")
print(f"{'K':>4} {'ρ(pred,exact)':>15} {'ρ(pred,pos)':>13} {'ρ(exact,pos)':>14} "
      f"{'std(D_n)':>10} {'std(KL_n)':>11}")

for K in (16, 32, 64):
    cfg = Config()
    cfg.cache.budget = 256
    cfg.memory.num_slots = K
    cfg = cfg.ablation(4)
    model, tok, mem = build(cfg)
    mem.eval()
    ck = Path(f"varikv/ckpt/k{K}_tier4.pt")
    if ck.exists():
        mem.load_state_dict(torch.load(ck, map_location=cfg.device)["memory"])

    sc = mem.scorer
    rec = []
    orig_score = sc.score

    def spy(memory, k, v, rel_pos, force_exact=False):
        # 精确 F（只为诊断，不改变实际行为）
        F_exact, aux = sc.exact(memory, k, v, rel_pos)
        # 预测器 F —— 这才是评测时真正用于驱逐的
        F_pred = sc.predicted(aux["evidence"], aux["expected_attn"], rel_pos,
                              sc.memory_summary(memory))
        e = F_exact.float().mean(dim=1).flatten().tolist()
        p = F_pred.float().mean(dim=1).flatten().tolist()
        pos = rel_pos.float().mean(dim=1).flatten().tolist()
        n = k.shape[-2]
        D_n = (aux["D"].float() * (n ** 2) / sc.v_scale.clamp_min(1e-6))
        KL_n = aux["KL"].float() / memory.d_z
        rec.append((p, e, pos, D_n.flatten().tolist(), KL_n.flatten().tolist()))
        return F_pred, {}

    sc.score = spy
    with torch.no_grad():
        for s in val:
            ctx, q, a = encode_sample(tok, s, cfg.device)
            mem.prefill(model, ctx)
    sc.score = orig_score

    pe, pp, ep, sd, sk = [], [], [], [], []
    for p, e, pos, D_n, KL_n in rec:
        m = min(len(p), len(e), len(pos))
        if m < 8:
            continue
        pe.append(spearman(p[:m], e[:m]))
        pp.append(spearman(p[:m], pos[:m]))
        ep.append(spearman(e[:m], pos[:m]))
        sd.append(st.pstdev(D_n))
        sk.append(st.pstdev(KL_n))
    f = lambda x: sum(x) / len(x) if x else float("nan")
    print(f"{K:>4} {f(pe):>15.3f} {f(pp):>13.3f} {f(ep):>14.3f} "
          f"{f(sd):>10.4f} {f(sk):>11.4f}")

    del model, mem
    torch.cuda.empty_cache()

print("\nρ(pred,exact) 低 = 摊销失败，驱逐拿到的是噪声；"
      "\nρ(pred,pos) 高 = 预测器退回到照位置排序，即变成一个更差的 recency。")
