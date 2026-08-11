"""驱逐判据的对照实验：F 到底该怎么用？

背景（2026-08-07）：秩目标修好摊销后 ρ(pred,exact)=+0.78，但 tier4 反而变差
（2.81→3.02），比朴素 recency 差 +0.157。所以问题在 **F 这个判据本身**。

本实验**绕开预测器、直接用 exact 打分**，把 F 拆开逐项检验：

  recency   位置越靠后越留            —— 参照基线
  F         D_n/σ_D + λ·KL_n/σ_KL     —— 现行判据（高分保留）
  -F        取反                      —— 检验符号约定是否反了
  D         只要失真项（λ=0）         —— 退化成 Expected Attention
  -D
  KL        只要惊讶项
  -KL
  F_lam3    λ=3，让 KL 主导
  F_lam0.1  λ=0.1，让 D 主导

理论上高 F = 压进记忆代价大 = 该留精确。但对**写入**成立的直觉未必对**驱逐**成立：
高 KL 的观测正因为会被强力写入记忆，反而可能适合降级；低 KL 的已被记忆覆盖，
留着是冗余。符号是否反了必须实测。

用法： python scratch_evict_variants.py <variant> [tier] [K] [per_level]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from varikv.config import Config
from varikv.train import build, encode_sample, forward_loss
from stage1 import data as stage1_data

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "F"
TIER = int(sys.argv[2]) if len(sys.argv) > 2 else 4
K = int(sys.argv[3]) if len(sys.argv) > 3 else 16
PER_LEVEL = int(sys.argv[4]) if len(sys.argv) > 4 else 20
CKDIR = sys.argv[5] if len(sys.argv) > 5 else "varikv/ckpt"

LAM = {"F": None, "-F": None, "F_lam3": 3.0, "F_lam0.1": 0.1}
BUFS = ("v_scale", "v_init", "d_std", "kl_std", "std_init")

cfg = Config()
cfg.cache.budget = 256
cfg.memory.num_slots = K
cfg = cfg.ablation(TIER)
model, tok, mem = build(cfg)
mem.eval()
ck = Path(CKDIR) / f"k{K}_tier{TIER}.pt"
if not ck.exists():
    ck = Path("varikv/ckpt") / f"k{K}_tier{TIER}.pt"
if ck.exists():
    mem.load_state_dict(torch.load(ck, map_location=cfg.device)["memory"])

sc = mem.scorer
orig_scores = mem._evict_scores


def patched(k, v, keep_from, n_real):
    """用 exact 打分构造各变体。recency 直接走原实现。"""
    if VARIANT == "random":
        # 关键对照：若 F ≈ random，说明 F 完全没有信号；
        # 若 F 明显好于 random 但不如 recency，说明 F 有信号只是弱于位置先验。
        g = torch.Generator(device="cpu").manual_seed(1234 + n_real)
        r = torch.rand(n_real, generator=g).to(k.device)
        return r.unsqueeze(0).expand(k.shape[0], n_real), {}
    if VARIANT == "recency":
        pos = torch.arange(n_real, device=k.device, dtype=torch.float32)
        return pos.unsqueeze(0).expand(k.shape[0], n_real), {}

    rel_pos = (torch.arange(n_real, device=k.device, dtype=torch.float32) / max(n_real, 1)
               ).view(1, 1, n_real).expand(k.shape[0], mem.n_groups, n_real)
    # 快照 running 缓冲：exact() 会就地更新它们，而正常评测路径根本不调 exact，
    # 不还原的话各变体之间的归一化基准会互相污染
    snap = {b: getattr(sc, b).clone() for b in BUFS if hasattr(sc, b)}
    F_exact, aux = sc.exact(mem.memory, k, v, rel_pos)
    for b, t in snap.items():
        getattr(sc, b).copy_(t)

    n = k.shape[-2]
    D_n = aux["D"].float() * (n ** 2) / sc.v_scale.clamp_min(1e-6)
    KL_n = aux["KL"].float() / mem.memory.d_z
    Dz = D_n / sc.d_std.clamp_min(1e-6)
    Kz = KL_n / sc.kl_std.clamp_min(1e-6)

    if VARIANT in ("F", "-F"):
        s = Dz + cfg.free_energy.lam * Kz
    elif VARIANT in ("F_lam3", "F_lam0.1"):
        s = Dz + LAM[VARIANT] * Kz
    elif VARIANT in ("D", "-D"):
        s = Dz
    elif VARIANT in ("KL", "-KL"):
        s = Kz
    else:
        raise ValueError(VARIANT)
    if VARIANT.startswith("-"):
        s = -s
    return s.mean(dim=1), {}


mem._evict_scores = patched

val = stage1_data.load("stage1/val.jsonl")
by = defaultdict(list)
for s in val:
    by[s.n_distract].append(s)
val = [s for lv in sorted(by) for s in by[lv][:PER_LEVEL]]

buckets = defaultdict(lambda: [0, 0.0])
with torch.no_grad():
    for s in val:
        ctx, q, a = encode_sample(tok, s, cfg.device)
        nll = forward_loss(model, mem, ctx, q, a).item()
        buckets[s.n_distract][0] += 1
        buckets[s.n_distract][1] += nll

evicted = [l for l in sorted(buckets) if l > 0]
tot_n = sum(buckets[l][0] for l in evicted)
tot = sum(buckets[l][1] for l in evicted) / max(tot_n, 1)
out = {"variant": VARIANT, "tier": TIER, "K": K, "nll_evicted": tot,
       "per_level": {str(l): buckets[l][1] / buckets[l][0] for l in sorted(buckets)}}
print(json.dumps(out))
