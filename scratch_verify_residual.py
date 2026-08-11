"""残差改造的正确性验证。四项必须全过才能开跑。

1. 记忆确实不进 cache（序列长度与基线一致），但吸收仍在发生
2. 门关到底（sigmoid→0）时**逐字等于基线** —— 这是「零成本退回」的硬保证
3. 门开大时输出**确实改变** —— 证明这条通路是活的（不是被静默旁路）
4. 梯度能同时到达 decoder 与 gate —— 否则训练又是「loss 正常、什么都没学」
"""
import sys
from pathlib import Path

import torch

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip                # noqa: E402
from data.load import load_dataset_all              # noqa: E402
from varikv.config import Config                    # noqa: E402
from varikv.memory import DistributionalMemory      # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
KW = dict(prefill_chunk_size=16000, do_score=True, chunk_ratio=0.3,
          window_size=4096, level="pair")


def build(kvt, residual, gate_val=None, train=False):
    m = ModelKVzip(MODEL, kvt, "fastkvzip")
    if kvt == "retain":
        return m, None
    cfg = Config()
    H, L = m.config.num_key_value_heads, m.config.num_hidden_layers
    hd = getattr(m.config, "head_dim",
                 m.config.hidden_size // m.config.num_attention_heads)
    mem = DistributionalMemory(2 * hd, cfg.memory, mode="dist",
                               n_groups=L * H).to(m.device, dtype=torch.float32)
    sd = torch.load(R / "varikv/ckpt_stage2b_retain/s2b_dist_k16.pt",
                    map_location=m.device)["memory"]
    mem.load_state_dict(sd, strict=False)      # ckpt 无 residual_gate，用初值
    if gate_val is not None:
        with torch.no_grad():
            mem.residual_gate.fill_(gate_val)
    mem.train() if train else mem.eval()
    for p in mem.parameters():
        p.requires_grad_(train)
    m.varikv_memory, m.varikv_M = mem, 16
    m.varikv_residual = residual
    m.varikv_train = train
    rot = getattr(m.model.model, "rotary_emb", None)
    m.varikv_inv_freq = rot.inv_freq.detach().clone()
    return m, mem


ds = None
res = {}
for tag, kvt, residual, gate in (
        ("基线 retain", "retain", False, None),
        ("残差·门关到底(-30)", "memory_retain", True, -30.0),
        ("残差·门开(+3)", "memory_retain", True, 3.0),
):
    m, mem = build(kvt, residual, gate)
    if ds is None:
        ds = load_dataset_all("scbench_kv", m.tokenizer, n_data=1)
    kv = m.prefill(ds[0]["context"], **KW)
    out = m.generate(ds[0]["question"][0], kv=kv)
    seq = kv.key_cache[0].shape[2]
    absorbed = getattr(kv, "stats", {}).get("absorbed", 0)
    res[tag] = (seq, out)
    print(f"[{tag}] cache序列={seq} 吸收={absorbed} 生成={out[:52]!r}")
    del m, kv
    torch.cuda.empty_cache()

base_seq, base_out = res["基线 retain"]
off_seq, off_out = res["残差·门关到底(-30)"]
on_seq, on_out = res["残差·门开(+3)"]
print()
print(f"1) 记忆不进 cache      : {'✓' if off_seq == base_seq else '✗'} "
      f"({off_seq} vs 基线 {base_seq})")
print(f"2) 门关→逐字等于基线   : {'✓' if off_out == base_out else '✗ 不等'}")
print(f"3) 门开→输出确实改变   : {'✓' if on_out != base_out else '✗ 通路是死的'}")

# ---- 4) 梯度 ----
m, mem = build("memory_retain", True, gate_val=-1.0, train=True)
ids = m.encode(ds[0]["context"])[:, :40000]
tgt = m.encode(ds[0]["context"])[:, 40000:40128]
with torch.no_grad():
    kv = m.prefill(ids, **KW)
out = m.model(tgt, past_key_values=kv)
loss = torch.nn.functional.cross_entropy(
    out.logits[:, :-1].float().reshape(-1, out.logits.size(-1)), tgt[:, 1:].reshape(-1))
loss.backward()
g_dec = max((p.grad.abs().max().item() for n, p in mem.named_parameters()
             if p.grad is not None and "decoder" in n), default=0.0)
g_enc = max((p.grad.abs().max().item() for n, p in mem.named_parameters()
             if p.grad is not None and "encoder" in n), default=0.0)
g_gate = mem.residual_gate.grad.abs().max().item() if mem.residual_gate.grad is not None else 0.0
print(f"4) 梯度 decoder={g_dec:.2e} encoder={g_enc:.2e} gate={g_gate:.2e}  "
      f"{'✓' if g_dec > 0 and g_gate > 0 else '✗ 有通路断了'}")
