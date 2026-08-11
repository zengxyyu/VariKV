"""K 增大时 F 的两个分量各自还剩多少区分度？

已定位的现象：K=16→64，自由能驱逐保留的位置从分散(0.57)塌向末尾(0.84)，
针的保留率 7.2%→0，即**退化成一个更差的 recency**。

待验机制：F = D/σ_D + λ·KL/σ_KL。槽越多，混合先验越容易把任何证据都解释掉
→ KL 对所有 i 都变小且彼此拉不开 → F 的排序被 D 主导。
（CLAUDE.md 已记过同类失效：归一化只按尺度不按**离散度**时，
  「F 的排序 99% 由 D 决定，与 KL 的秩相关只有 0.09」。）

所以直接量：各分量的离散度、以及 F 分别与 D / KL / 位置的秩相关。
"""
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
print(f"level={LEVEL}  n={N}   （tier4 = free_energy + point）\n")
print(f"{'K':>4} {'std(D_n)':>10} {'std(KL_n)':>11} {'CV(KL_n)':>10} "
      f"{'ρ(F,D)':>8} {'ρ(F,KL)':>9} {'ρ(F,pos)':>9}")

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

    rec = []
    orig = mem._evict_scores

    def spy(k, v, keep_from, n_real):
        scores, aux = orig(k, v, keep_from, n_real)
        if aux and "D" in aux and "KL" in aux:
            sc = mem.scorer
            D = aux["D"].float()
            kl = aux["KL"].float()
            n = k.shape[-2]
            D_n = D * (n ** 2) / sc.v_scale.clamp_min(1e-6)
            KL_n = kl / mem.memory.d_z
            rec.append((D_n.flatten().tolist(), KL_n.flatten().tolist(),
                        scores.float().flatten().tolist(),
                        mem.token_pos.float().tolist() if mem.token_pos is not None else None))
        return scores, aux

    mem._evict_scores = spy
    with torch.no_grad():
        for s in val:
            ctx, q, a = encode_sample(tok, s, cfg.device)
            mem.prefill(model, ctx)
    mem._evict_scores = orig

    if not rec:
        print(f"{K:>4}   （没抓到打分调用）")
        continue

    import statistics as st
    sd, sk, cv, rfd, rfk, rfp = [], [], [], [], [], []
    for D_n, KL_n, F, pos in rec:
        m = min(len(D_n), len(KL_n), len(F))
        if m < 8:
            continue
        D_n, KL_n, F = D_n[:m], KL_n[:m], F[:m]
        sd.append(st.pstdev(D_n))
        sk.append(st.pstdev(KL_n))
        mk = sum(KL_n) / m
        cv.append(st.pstdev(KL_n) / (abs(mk) + 1e-12))
        rfd.append(spearman(F, D_n))
        rfk.append(spearman(F, KL_n))
        if pos and len(pos) >= m:
            rfp.append(spearman(F, pos[:m]))
    f = lambda x: sum(x) / len(x) if x else float("nan")
    print(f"{K:>4} {f(sd):>10.4f} {f(sk):>11.4f} {f(cv):>10.4f} "
          f"{f(rfd):>8.3f} {f(rfk):>9.3f} {f(rfp):>9.3f}")

    del model, mem
    torch.cuda.empty_cache()

print("\nρ(F,pos) 接近 +1 表示 F 的排序已经等价于「越晚越该留」，即退化成 recency。")
