"""Stage 2b 移植验证 —— 目标模型 Qwen2.5-7B-Instruct-1M + 真实 fastkvzip 门控。

1.5B 上验证过的东西不能直接外推：kv_head 从 2 变 4（G 从 56 变 112），
而且 1.5B 没有发布的门控权重、只能用 expect 顶替。这里两个都换成真的。
"""
import sys
from pathlib import Path
import torch

ROOT = Path("/home/ubuntu/zxy/vlm-memory")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip
from varikv.config import Config
from varikv.memory import DistributionalMemory

MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
GATE = sys.argv[1] if len(sys.argv) > 1 else "fastkvzip"
LEVEL = sys.argv[2] if len(sys.argv) > 2 else "pair"
M = int(sys.argv[3]) if len(sys.argv) > 3 else 16

ctx = "Log:\n" + "\n".join(f'SET s_{i:05d} = "v-{i*13%9973}"' for i in range(2500))
Q = "What is the value of s_00042?"
KW = dict(prefill_chunk_size=4096, do_score=True, window_size=256, level=LEVEL)


def build(kv_type, with_mem):
    m = ModelKVzip(MODEL, kv_type=kv_type, gate_path_or_name=GATE)
    if with_mem:
        cfg = Config(); cfg.memory.num_slots = M
        H = m.config.num_key_value_heads; L = m.config.num_hidden_layers
        hd = getattr(m.config, "head_dim",
                     m.config.hidden_size // m.config.num_attention_heads)
        mem = DistributionalMemory(2 * hd, cfg.memory, mode="dist").to(
            m.device, dtype=m.dtype)
        mem.reset(1, L * H, device=m.device, dtype=m.dtype)
        m.varikv_memory = mem; m.varikv_M = M
        rot = getattr(m.model.model, "rotary_emb", None)
        m.varikv_inv_freq = rot.inv_freq.detach().clone() if rot else None
    return m


def layout_ok(kv):
    for l in range(kv.n_layers):
        real = kv.key_cache[l].shape[0]
        if int(kv.info["len_k"][l].sum()) != real: return False
        if int(kv.info["cu_len_k"][l][-1]) != real: return False
        pt = getattr(kv, "pos_track", [None] * kv.n_layers)[l]
        if pt is not None and pt.numel() != real: return False
    return True


print(f"模型 {MODEL}\n门控 {GATE}  level {LEVEL}  M {M}")
res = {}
for tag, kt, wm in (("evict原生", "evict", False),
                    ("memory关吸收", "memory", False),
                    ("memory开吸收", "memory", True)):
    m = build(kt, wm)
    cfg_ = m.config
    if tag == "evict原生":
        print(f"layers={cfg_.num_hidden_layers} kv_heads={cfg_.num_key_value_heads} "
              f"→ G={cfg_.num_hidden_layers*cfg_.num_key_value_heads}")
        print(f"上下文 {len(m.tokenizer(ctx).input_ids)} token\n")
    for r in (0.3, 0.1):
        kv = m.prefill(ctx, chunk_ratio=r, **KW)
        # 必须统计**全部层**：level="pair" 是跨层全局阈值，各层预算会被重新
        # 分配，只看 layer 0 会把「预算重分布」误读成「总量变了」。
        tot = float(sum(int(x.sum()) for x in kv.info["len_k"]))
        out = m.generate(Q, kv=kv)
        for q2 in ["What is the value of s_00100?", "How many entries?"]:
            m.generate(q2, kv=kv)
        res.setdefault(r, {})[tag] = (tot, out, layout_ok(kv),
                                      getattr(kv, "stats", {}))
    del m; torch.cuda.empty_cache()

for r in (0.3, 0.1):
    print(f"\n--- ratio {r} ---")
    base = res[r]["evict原生"][0]
    for tag in ("evict原生", "memory关吸收", "memory开吸收"):
        tot, out, ok, st = res[r][tag]
        d = f"{(tot-base)/base*100:+.2f}%" if base else "-"
        print(f"  {tag:14s} 总KV {tot:9.0f} ({d:>8})  布局{'✓' if ok else '✗'}  "
              f"生成 {out[:44]!r}")
        if st.get("budget_shortfall"):
            print(f"                 缺口 {st['budget_shortfall']}")
    same = res[r]["evict原生"][1] == res[r]["memory关吸收"][1]
    print(f"  关吸收与原生输出一致: {'✓' if same else '✗'}")
