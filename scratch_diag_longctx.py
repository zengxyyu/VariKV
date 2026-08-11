"""诊断：dist 档在长上下文上崩溃（mf/kv/prefix_suffix 相对性能 0.00~1.20）。

关键线索：崩溃随上下文长度单调恶化 —— squad(203) 97.6 → vt(125k) 54.5 →
kv(169k) 0.29，而 point 档在同样长度下仍有 60。上下文越长 → 吸收轮数越多，
所以嫌疑集中在 dist 特有的方差/精度递推上（point 档 tau 恒为 1，不递推）。

训练只在 ≤32k 上做过，169k 是 5 倍外推。

逐 chunk 记录记忆状态与读出 KV 的健康度，并与真实 KV 的量级对比 ——
若读出的等效 KV 范数远大于真实 KV，它就会靠幅度而非内容抢占注意力。
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip            # noqa: E402
from data.load import load_dataset_all          # noqa: E402
from varikv.config import Config                # noqa: E402
from varikv.memory import DistributionalMemory  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "dist"
DS = sys.argv[2] if len(sys.argv) > 2 else "scbench_kv"
MAXCTX = int(sys.argv[3]) if len(sys.argv) > 3 else 0     # 0 = 全长

ck = ROOT / f"varikv/ckpt_stage2b_matched/s2b_{MODE}_k16.pt"
m = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M", "memory", "fastkvzip")
cfg = Config()
H, L = m.config.num_key_value_heads, m.config.num_hidden_layers
hd = getattr(m.config, "head_dim", m.config.hidden_size // m.config.num_attention_heads)
mem = DistributionalMemory(2 * hd, cfg.memory, mode=MODE).to(m.device, dtype=torch.float32)
mem.load_state_dict(torch.load(ck, map_location=m.device)["memory"])
mem.eval()
m.varikv_memory, m.varikv_M = mem, 16
rot = getattr(m.model.model, "rotary_emb", None)
m.varikv_inv_freq = rot.inv_freq.detach().clone()

ds = load_dataset_all(DS, m.tokenizer, n_data=1)
ctx = ds[0]["context"]
ids = m.encode(ctx)
if MAXCTX:
    ids = ids[:, :MAXCTX]
print(f"\nmode={MODE} dataset={DS} ctx={ids.shape[1]} token")

import attention.memcache as MC
rec = []
orig = MC.MemoryEvictCache._refresh_memory


def spy(self, layer_idx):
    orig(self, layer_idx)
    if layer_idx != 0:
        return
    gs = slice(0, self.n_heads_kv)
    mu = self.mem.mu[:, gs].float()
    lv = self.mem.logvar[:, gs].float()
    lo, hi = self.mem.cfg.logvar_min, self.mem.cfg.logvar_max
    # 读出的等效 KV vs 真实 KV 的量级
    k_all = self.key_cache[0].float()
    cu = self.info["cu_len_k"][0]
    v_all = self.value_cache[0].float()
    memk = torch.cat([k_all[int(cu[h]): int(cu[h]) + self.M] for h in range(self.n_heads_kv)])
    realk = torch.cat([k_all[int(cu[h]) + self.M: int(cu[h]) + int(self.info["len_k"][0][h])]
                       for h in range(self.n_heads_kv)])
    memv = torch.cat([v_all[int(cu[h]): int(cu[h]) + self.M] for h in range(self.n_heads_kv)])
    realv = torch.cat([v_all[int(cu[h]) + self.M: int(cu[h]) + int(self.info["len_k"][0][h])]
                       for h in range(self.n_heads_kv)])
    rec.append(dict(
        n_seen=self._seen_real[0],
        mu_absmax=mu.abs().max().item(),
        lv_mean=lv.mean().item(), lv_min=lv.min().item(), lv_max=lv.max().item(),
        at_lo=(lv <= lo + 1e-3).float().mean().item(),
        at_hi=(lv >= hi - 1e-3).float().mean().item(),
        nan=int(torch.isnan(mu).sum() + torch.isnan(lv).sum()),
        memk_norm=memk.norm(dim=-1).mean().item(),
        realk_norm=realk.norm(dim=-1).mean().item() if realk.numel() else float("nan"),
        memv_norm=memv.norm(dim=-1).mean().item(),
        realv_norm=realv.norm(dim=-1).mean().item() if realv.numel() else float("nan"),
    ))


MC.MemoryEvictCache._refresh_memory = spy
with torch.no_grad():
    m.prefill(ids, prefill_chunk_size=16000, do_score=True, chunk_ratio=0.3,
              window_size=4096, level="pair")
MC.MemoryEvictCache._refresh_memory = orig

print(f"\n{'轮':>3} {'n_seen':>8} {'记忆K':>9} {'真实K':>9} {'K比值':>7} "
      f"{'记忆V':>10} {'真实V':>9} {'V比值':>8}")
for i, r in enumerate(rec):
    ratio = r["memk_norm"] / r["realk_norm"] if r["realk_norm"] == r["realk_norm"] else float("nan")
    vr = r["memv_norm"]/r["realv_norm"]
    print(f"{i:>3} {r['n_seen']:>8} {r['memk_norm']:>9.3f} {r['realk_norm']:>9.2f} "
          f"{ratio:>7.3f} {r['memv_norm']:>10.3f} {r['realv_norm']:>9.3f} {vr:>8.2f}")
