"""确认 ρ(pred,exact)<0，并区分两种成因。

上一轮测到预测器与精确 F **反相关**（-0.37~-0.47）。下结论前排除两件事：

  (1) 测量污染：exact() 会就地更新 v_scale/d_std/kl_std 三个 running 缓冲，
      而 F = D_n/d_std + λ·KL_n/kl_std 的**排序**依赖这两个分母的比例。
      评测时正常路径根本不调 exact()，所以我的探针本身会改变被测量。
      → 每次调用前后快照并还原缓冲。

  (2) 训练/评测分布漂移：预测器是在 max_train_context=4096（8 块）上蒸馏的，
      评测却跑到 34k（67 块），记忆状态完全不同。
      → 同时在 4096 和全长上测，若短上下文为正、长上下文为负，就是迁移失败，
        而不是没学会。
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
K = int(sys.argv[2]) if len(sys.argv) > 2 else 16
CKPT = sys.argv[3] if len(sys.argv) > 3 else None


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


cfg = Config()
cfg.cache.budget = 256
cfg.memory.num_slots = K
cfg = cfg.ablation(4)
model, tok, mem = build(cfg)
mem.eval()
ck = Path(CKPT) if CKPT else Path(f"varikv/ckpt/k{K}_tier4.pt")
if ck.exists():
    mem.load_state_dict(torch.load(ck, map_location=cfg.device)["memory"])
    print(f"loaded {ck}")
sc = mem.scorer

val = [s for s in stage1_data.load("stage1/val.jsonl") if s.n_distract == LEVEL][:N]
print(f"K={K} level={LEVEL} n={N}\n")
print(f"{'上下文长度':>12} {'块数':>5} {'ρ(pred,exact)':>15} {'ρ(pred,pos)':>13} "
      f"{'ρ(exact,pos)':>14}")

BUFS = ("v_scale", "v_init", "d_std", "kl_std", "std_init")

for max_ctx in (4096, 8192, 16384, 0):
    rec = []
    orig_score = sc.score

    def spy(memory, k, v, rel_pos, force_exact=False):
        # 快照 running 缓冲，算完精确 F 后还原 —— 探针不得改变被测系统
        snap = {b: getattr(sc, b).clone() for b in BUFS if hasattr(sc, b)}
        F_exact, aux = sc.exact(memory, k, v, rel_pos)
        for b, t in snap.items():
            getattr(sc, b).copy_(t)
        F_pred = sc.predicted(aux["evidence"], aux["expected_attn"], rel_pos,
                              sc.memory_summary(memory))
        rec.append((F_pred.float().mean(1).flatten().tolist(),
                    F_exact.float().mean(1).flatten().tolist(),
                    rel_pos.float().mean(1).flatten().tolist()))
        return F_pred, {}

    sc.score = spy
    n_chunks = []
    with torch.no_grad():
        for s in val:
            ctx, q, a = encode_sample(tok, s, cfg.device)
            sub = ctx[:, :max_ctx] if max_ctx else ctx
            n_chunks.append(-(-sub.shape[1] // cfg.cache.prefill_chunk))
            mem.prefill(model, sub)
    sc.score = orig_score

    pe, pp, ep = [], [], []
    for p, e, pos in rec:
        m = min(len(p), len(e), len(pos))
        if m < 8:
            continue
        pe.append(spearman(p[:m], e[:m]))
        pp.append(spearman(p[:m], pos[:m]))
        ep.append(spearman(e[:m], pos[:m]))
    f = lambda x: sum(x) / len(x) if x else float("nan")
    lab = f"{max_ctx}" if max_ctx else "全长(13.8k)"
    print(f"{lab:>12} {int(f(n_chunks)):>5} {f(pe):>15.3f} {f(pp):>13.3f} {f(ep):>14.3f}")

print("\n训练用 max_train_context=4096。若 4096 处也为负 → 预测器压根没学会；")
print("若 4096 为正而长上下文转负 → 蒸馏有效但迁移失败。")
