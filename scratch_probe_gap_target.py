"""判定 `--obj gap` 目标是否退化：把达成的 MSE 和平凡解 m≡0 的 MSE 直接比。

为什么必须做：训练日志里 gap 目标的 loss 收到 0.003、|g| 1e-04、门被训到低于初值，
看着"收敛得很好"。但 `m → 0` 就在解空间内（loss = MSE(g·m, o_full − o_pruned)，
memcache_retain.py:295），所以 0.003 完全可能就是平凡解的值。这和当年 F 预测器被
Huber 压成常数（loss 0.0419 而恒输出 0 是 0.0421）是同一类陷阱。

三个量，逐 (sample, layer)：
  triv  = mean(tgt²)                      —— m≡0 的 MSE，平凡解基准
  ach   = mean((m − tgt)²)                —— 实际达成
  R_opt = 只重调每个 head 的门能达到的最大相对下降
          = Σ_h [<m̂_h,tgt_h>² / ‖m̂_h‖²] / Σ_h ‖tgt_h‖²
          其中 m̂ = m/σ(gate) 是未门控的读出

R_opt 是本探针的关键数字，它把两件事分开：
  R_opt ≈ 0     ⇒ 读出内容本身没有信息，关门是最优解，目标在此参数化下确实退化
  R_opt 明显 >0 ⇒ 内容有信息但训练/门没利用上，是优化问题而非目标问题
按 head 拟合而不是按层，是因为门本来就是 per-head 的（这样对方法最公平）。
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip                        # noqa: E402
from varikv.config import Config                            # noqa: E402
from varikv.memory import DistributionalMemory              # noqa: E402
from attention.memcache_retain import MemoryRetainCache     # noqa: E402


def build(model_name, gate, num_slots, mode, ckpt):
    """与 scratch_stage2b_train.py:build 同一条路径，只是加载已训好的权重。"""
    m = ModelKVzip(model_name, kv_type="memory_retain", gate_path_or_name=gate)
    cfg = Config()
    cfg.memory.num_slots = num_slots
    H = m.config.num_key_value_heads
    L = m.config.num_hidden_layers
    hd = getattr(m.config, "head_dim",
                 m.config.hidden_size // m.config.num_attention_heads)
    mem = DistributionalMemory(2 * hd, cfg.memory, mode=mode,
                               n_groups=L * H).to(m.device, dtype=torch.float32)
    mem.reset(1, L * H, device=m.device, dtype=torch.float32)
    if ckpt:
        sd = torch.load(ckpt, map_location=m.device)["memory"]
        info = mem.load_state_dict(sd, strict=False)
        print(f"[ckpt] {ckpt}\n       missing={list(info.missing_keys)} "
              f"unexpected={list(info.unexpected_keys)}")
    m.varikv_memory = mem
    m.varikv_M = num_slots
    m.varikv_train = True          # 绕开 prefill 的 inference_mode（我们自己套 no_grad）
    m.varikv_residual = True
    m.varikv_detach_readback = False
    rot = getattr(m.model.model, "rotary_emb", None)
    m.varikv_inv_freq = rot.inv_freq.detach().clone() if rot else None
    return m, mem, L, H


# ---------------------------------------------------------------- 探针挂钩
_REC = []          # [(layer, triv, ach, num_per_head, den_per_head, tgt_sq_per_head)]

_orig_gap = MemoryRetainCache._attn_gap
_orig_res = MemoryRetainCache.memory_residual


def _patched_gap(self, query_states, layer_idx):
    t = _orig_gap(self, query_states, layer_idx)
    self._probe_tgt = t
    return t


def _patched_res(self, query_states, layer_idx):
    self._probe_tgt = None
    out = _orig_res(self, query_states, layer_idx)      # [B,T,H*Gq*d]，已乘门
    tgt = self._probe_tgt
    if tgt is not None:
        H, d = self.n_heads_kv, self.head_dim
        B, T, F = out.shape
        Gq = F // (H * d)
        # forward 是 m.permute(0,3,1,2,4).reshape(B,T,H*Gq*d)，这里逆回去
        m = out.view(B, T, H, Gq, d).permute(0, 2, 3, 1, 4).float()   # [B,H,Gq,T,d]
        tgt = tgt.float()
        g = torch.sigmoid(
            self.mem.residual_gate[layer_idx * H:(layer_idx + 1) * H]).float()
        m_hat = m / g.view(1, H, 1, 1, 1).clamp_min(1e-8)             # 去掉门
        dims = (0, 2, 3, 4)                                          # 保留 head 维
        _REC.append((
            layer_idx,
            tgt.pow(2).mean().item(),
            (m - tgt).pow(2).mean().item(),
            (m_hat * tgt).sum(dim=dims).cpu(),
            m_hat.pow(2).sum(dim=dims).cpu(),
            tgt.pow(2).sum(dim=dims).cpu(),
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--gate", default="fastkvzip")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", default="dist", choices=["point", "dist"])
    ap.add_argument("--num_slots", type=int, default=16)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--chunk", type=int, default=16000)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--level", default="pair")
    ap.add_argument("--target_len", type=int, default=128)
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--max_ctx", type=int, default=32000)
    args = ap.parse_args()

    MemoryRetainCache._attn_gap = _patched_gap
    MemoryRetainCache.memory_residual = _patched_res

    m, mem, L, H = build(args.model, args.gate, args.num_slots, args.mode,
                         args.ckpt)
    gsig = torch.sigmoid(mem.residual_gate.detach()).cpu()
    print(f"[gate] σ 均值 {gsig.mean():.4f}  最大 {gsig.max():.4f}  "
          f">0.1 占比 {(gsig > 0.1).float().mean():.2f}  (共 {gsig.numel()} 组)")

    from data.load import load_fineweb
    docs = load_fineweb("fineweb_10k")[:args.n_samples]

    per_layer = {}
    for si, d in enumerate(docs):
        ids = m.encode(d["context"])[0].tolist()
        ctx_ids = ids[-(args.max_ctx + args.target_len):-args.target_len]
        tgt_ids = ids[-args.target_len:]
        _REC.clear()
        mem.reset(1, L * H, device=m.device, dtype=torch.float32)
        with torch.no_grad():
            kv = m.prefill(torch.tensor([ctx_ids], device=m.device),
                           prefill_chunk_size=args.chunk, do_score=True,
                           chunk_ratio=args.ratio, window_size=args.window,
                           level=args.level)
            kv.collect_residual_loss = True
            kv.residual_losses = []
            m.model.model(torch.tensor([tgt_ids], device=m.device),
                          past_key_values=kv)
            kv.collect_residual_loss = False
        ach_reported = (torch.stack(kv.residual_losses).mean().item()
                        if kv.residual_losses else float("nan"))
        print(f"\n样本 {si}: ctx {len(ctx_ids)} tok, 记录 {len(_REC)} 层, "
              f"训练口径 loss {ach_reported:.6f}")
        for layer, triv, ach, num, den, tsq in _REC:
            per_layer.setdefault(layer, []).append((triv, ach, num, den, tsq))
        del kv
        torch.cuda.empty_cache()

    # ------------------------------------------------------------ 汇总
    print("\n" + "=" * 78)
    print(f"{'layer':>5} {'triv=mean(tgt²)':>16} {'达成 MSE':>12} "
          f"{'达成/平凡':>10} {'R_opt(%)':>9}")
    tot_t = tot_a = 0.0
    tot_num2den = tot_tsq = 0.0
    for layer in sorted(per_layer):
        recs = per_layer[layer]
        triv = sum(r[0] for r in recs) / len(recs)
        ach = sum(r[1] for r in recs) / len(recs)
        num = torch.stack([r[2] for r in recs]).sum(0)
        den = torch.stack([r[3] for r in recs]).sum(0)
        tsq = torch.stack([r[4] for r in recs]).sum(0)
        expl = (num.pow(2) / den.clamp_min(1e-20)).sum().item()
        r_opt = 100.0 * expl / max(tsq.sum().item(), 1e-20)
        tot_t += triv; tot_a += ach
        tot_num2den += expl; tot_tsq += tsq.sum().item()
        print(f"{layer:>5} {triv:>16.6f} {ach:>12.6f} "
              f"{ach/max(triv,1e-20):>10.4f} {r_opt:>9.3f}")
    n = len(per_layer)
    print("-" * 78)
    print(f"{'全层':>5} {tot_t/n:>16.6f} {tot_a/n:>12.6f} "
          f"{(tot_a/n)/max(tot_t/n,1e-20):>10.4f} "
          f"{100.0*tot_num2den/max(tot_tsq,1e-20):>9.3f}")
    print("=" * 78)
    print("判读：达成/平凡 ≈ 1.00 ⇒ 与 m≡0 无异，loss 数值本身没有意义。")
    print("      R_opt ≈ 0 ⇒ 读出内容无信息，关门是最优解（目标在此参数化下退化）。")
    print("      R_opt 明显 >0 ⇒ 内容有信息，是门/优化没利用上，不是目标退化。")


if __name__ == "__main__":
    main()
