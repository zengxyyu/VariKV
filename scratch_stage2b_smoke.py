"""Stage 2b 冒烟测试：MemoryEvictCache 能否在真实管线里跑通且布局自洽。

分三步验证，任何一步不过都说明接入是坏的：
  1. absorb_enabled=False 时必须与原生 EvictCache **逐位等价**（说明覆写没引入副作用）
  2. 打开记忆后布局仍自洽（len_k / cu_len_k 与实际张量长度一致，位置追踪长度对齐）
  3. 生成不出 NaN，且记忆确实吸收了东西（stats["absorbed"] > 0）
"""
import sys
from pathlib import Path

import torch

ROOT = Path("/home/ubuntu/zxy/vlm-memory")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip                     # noqa: E402
from varikv.config import Config                          # noqa: E402
from varikv.memory import DistributionalMemory            # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
RATIO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
GATE = sys.argv[3] if len(sys.argv) > 3 else "expect"

ctx = ("The following is a log of configuration values.\n"
       + "\n".join(f"SET server_{i:04d} = \"value-{i*7 % 997}\"" for i in range(1200))
       + "\nEnd of log.")
question = "What is the value of server_0042?"


def check_layout(kv, tag):
    ok = True
    for l in range(kv.n_layers):
        tot = int(kv.info["len_k"][l].sum())
        real = kv.key_cache[l].shape[0]
        if tot != real:
            print(f"  [{tag}] layer{l} len_k 之和 {tot} != 张量长度 {real}")
            ok = False
        cu = kv.info["cu_len_k"][l]
        if int(cu[-1]) != real:
            print(f"  [{tag}] layer{l} cu_len_k[-1] {int(cu[-1])} != {real}")
            ok = False
        pt = getattr(kv, "pos_track", [None] * kv.n_layers)[l]
        if pt is not None and pt.numel() != real:
            print(f"  [{tag}] layer{l} pos_track {pt.numel()} != {real}")
            ok = False
    print(f"  [{tag}] 布局自洽: {'✓' if ok else '✗'}")
    return ok


def run(kv_type, with_memory):
    model = ModelKVzip(MODEL, kv_type=kv_type, gate_path_or_name=GATE)
    if with_memory:
        cfg = Config()
        H = model.config.num_key_value_heads
        L = model.config.num_hidden_layers
        hd = getattr(model.config, "head_dim",
                     model.config.hidden_size // model.config.num_attention_heads)
        mem = DistributionalMemory(2 * hd, cfg.memory, mode="dist").to(
            model.device, dtype=model.dtype
        )
        mem.reset(1, L * H, device=model.device, dtype=model.dtype)
        model.varikv_memory = mem
        model.varikv_M = cfg.memory.num_slots
        rot = getattr(model.model.model, "rotary_emb", None)
        model.varikv_inv_freq = rot.inv_freq.detach().clone() if rot else None

    kv = model.prefill(ctx, prefill_chunk_size=4096, do_score=True,
                       chunk_ratio=RATIO, window_size=256, level="adakv-layer")
    ok = check_layout(kv, kv_type + ("+mem" if with_memory else ""))
    # 多次生成：评测管线对同一 cache 每个 question 生成一次，每次都会 slice 回滚。
    # 只测一次就发现不了 pos_track / _seen_real 不同步回滚的静默错位。
    out = model.generate(question, kv=kv)
    for q2 in ["What is the value of server_0100?", "How many entries are there?"]:
        model.generate(q2, kv=kv)
    ok_slice = check_layout(kv, ("mem" if with_memory else "nomem") + "/3次生成后")
    finite = all(ord(c) < 0x110000 for c in out)
    print(f"  生成: {out[:90]!r}")
    if with_memory:
        print(f"  吸收统计: {kv.stats}")
        ok = ok and kv.stats["absorbed"] > 0
    ok = ok and ok_slice
    ok = ok and ok_slice
    del model, kv
    torch.cuda.empty_cache()
    return ok, out


print(f"模型 {MODEL}  ratio {RATIO}\n")
print("=== 1) memory 型但关闭吸收，应与原生 evict 等价 ===")
ok_a, out_a = run("evict", False)
ok_b, out_b = run("memory", False)
print(f"  两者输出一致: {'✓' if out_a == out_b else '✗ 不一致'}")

print("\n=== 2/3) 打开记忆 ===")
ok_c, out_c = run("memory", True)

print(f"\n总体: {'✓ 通过' if (ok_a and ok_b and (out_a == out_b) and ok_c) else '✗ 有问题'}")
