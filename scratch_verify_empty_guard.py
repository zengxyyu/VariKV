"""P0-A 自检：空记忆的残差读出必须精确为零，且吸收之后必须非零。"""
import sys
from pathlib import Path
import torch

ROOT = Path("/home/ubuntu/zxy/vlm-memory")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))
from varikv.config import Config
from varikv.memory import DistributionalMemory
from attention.memcache_retain import MemoryRetainCache

cfg = Config(); cfg.memory.num_slots = 16
d_head, H, L = 128, 4, 4
mem = DistributionalMemory(2 * d_head, cfg.memory, mode="dist", n_groups=L * H)
mem.reset(1, L * H, dtype=torch.float32)


class FakeCfg:
    num_key_value_heads = H
    num_attention_heads = H * 7
    hidden_size = H * 7 * d_head
    num_hidden_layers = L
    head_dim = d_head


class FakeModel:
    config = FakeCfg()


kv = MemoryRetainCache.__new__(MemoryRetainCache)   # 绕过 __init__ 的模型依赖
kv.mem = mem; kv.M = 16; kv.n_heads_kv = H; kv.head_dim = d_head
kv.inv_freq = None; kv.readout_mode = "normal"; kv.residual_mode = True
kv.collect_residual_loss = False; kv._absorbed_upto = 0
kv.n_layers = L

q = torch.randn(1, H * 7, 5, d_head)
out0 = kv.memory_residual(q, 0)
assert out0.shape == (1, 5, H * 7 * d_head), out0.shape
assert out0.abs().max().item() == 0.0, f"空记忆仍在注入：max|out|={out0.abs().max().item()}"
print(f"✓ 空记忆返回全零，形状 {tuple(out0.shape)}")

kv._absorbed_upto = 100                              # 假装吸收过
out1 = kv.memory_residual(q, 0)
assert out1.abs().max().item() > 0.0, "吸收后仍为零，guard 过紧"
print(f"✓ 吸收后非零，max|out|={out1.abs().max().item():.4e}")
print("P0-A 自检通过")
