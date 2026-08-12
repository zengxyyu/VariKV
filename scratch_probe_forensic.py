"""learned memory vs 质心：信息在哪一层丢的？（免训练取证）

**为什么需要它。** Retr.Prefix-Suffix 上，同样的被驱逐 KV：
    丢弃（基线）        8.60
    质心 K=16（免训练）  10.80  (+2.20)
    质心 K=1024         12.20  (+3.60★)
    learned VariKV K=16  8.00  (−0.60)
⇒ 信息不是不可恢复，是 **learned pathway 把它丢了或用错了**。但 learned pathway
有至少三处可能丢信息，这个脚本定位是哪一处：

    [k;v]∈R^256  →① encoder 压到 d_z=64  →② 混进 16 个槽  →③ decoder 解回 R^256
                 →④ 槽内单独 softmax 的加法残差读出

**关键陷阱（GPT 的原始方案会踩）：两个系统的 key 存在不同的旋转帧里。**
`memcache_retain` 写入时 `inverse_rope`（存无位置帧）、读出时 `apply_rope` 到槽的
位置质心；`centroid` 默认 `rope_mode="post"` 完全不旋转。实测真实 `inv_freq` 最快
分量 1.0 rad/token，两帧相差 74–93%，所以**直接比 `cos(k_i, k̂_j)` 测到的是旋转
而不是内容**。本脚本一律在**读出帧**里比较 —— 也就是推理时真实发生的那个帧。

三个指标（都在读出帧）：

  A. 可寻址性  max_j cos(k_i, k̂_j)  —— 被驱逐的 key 还能不能在记忆里找到近邻
  B. 打分保持  对真实 query，比较 top-相关被驱逐 token 的打分能否在记忆里复现
     具体：ρ_spearman( max_j(a·k̂_j 属于簇 j 的成员), a·k_i ) 的替代 ——
     直接看记忆产生的注意力分布与真实被驱逐集合注意力分布的重叠
  C. value 方向  cos( m(q), o_E(q) )  —— 寻址对了但内容方向错？

用法：CUDA_VISIBLE_DEVICES=0 .venv/bin/python scratch_probe_forensic.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip                     # noqa: E402
from attention.kvcache import RetainCache                # noqa: E402
from data.load import load_dataset_all                   # noqa: E402
from data.wrapper import DataWrapper, get_query          # noqa: E402
from varikv.rope import cos_sin_at, apply_rope           # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
CKPT = "varikv/ckpt_kl/s2b_dist_k16.pt"
RATIO, CHUNK, WINDOW, LEVEL = 0.1, 16000, 4096, "pair"
LAYERS = (0, 9, 14, 21, 26)

_Q = {}
_orig = RetainCache.prepare


def _p(self, q, k, v, l):
    _Q[l] = q.detach().clone()
    return _orig(self, q, k, v, l)


def get_valid(kv, l, n):
    v = kv._get_valid(l, n)
    while v.dim() > 2:
        v = v.squeeze(0)
    return v.bool()


@torch.no_grad()
def main():
    ds_name = sys.argv[1] if len(sys.argv) > 1 else "scbench_prefix_suffix"
    RetainCache.prepare = _p
    m = ModelKVzip(MODEL, kv_type="memory_retain", gate_path_or_name="fastkvzip")

    # 装上 learned memory（与评测同一套构建流程）
    from varikv.config import Config
    from varikv.memory import DistributionalMemory
    ck = torch.load(ROOT / CKPT, map_location=m.device)
    cfg = Config(); cfg.memory.num_slots = ck["num_slots"]
    H = m.config.num_key_value_heads; L = m.config.num_hidden_layers
    d = getattr(m.config, "head_dim", m.config.hidden_size // m.config.num_attention_heads)
    mem = DistributionalMemory(2 * d, cfg.memory, mode=ck["mode"],
                               n_groups=L * H).to(m.device, dtype=torch.float32)
    mem.load_state_dict(ck["memory"])
    m.varikv_memory = mem; m.varikv_M = ck["num_slots"]
    m.varikv_residual = True
    rot = getattr(m.model.model, "rotary_emb", None)
    m.varikv_inv_freq = rot.inv_freq.detach().clone()
    inv_freq = m.varikv_inv_freq

    ds = load_dataset_all(ds_name, m.tokenizer)
    dw = DataWrapper(ds_name, ds, m)
    q_ids = m.apply_template(get_query("qa", list(ds[0]["question"])[0])).to(m.device)
    kv = dw.prefill_context(0, prefill_chunk=CHUNK, window_size=WINDOW,
                            chunk_ratio=RATIO, level=LEVEL)
    _Q.clear()
    m.model(q_ids, past_key_values=kv)
    S = kv.key_cache[0].shape[2]
    M = ck["num_slots"]

    rows = []
    for l in LAYERS:
        valid = get_valid(kv, l, S).to(m.device)
        kall = kv.key_cache[l][0].float()
        vall = kv.value_cache[l][0].float()
        T = _Q[l].shape[2]
        G = _Q[l].shape[1] // H
        # learned 槽 → 读出帧（apply_rope 到位置质心），与推理时一致
        gs = slice(l * H, (l + 1) * H)
        kv._swap_in(gs)
        eff = mem.read().reshape(1, H, -1, 2 * d)[:, :, :M]
        pos = mem.pos.detach().clone().reshape(1, H, -1)[:, :, :M].reshape(H, M)
        kv._swap_out(gs)
        cos_, sin_ = cos_sin_at(inv_freq, pos, dtype=eff.dtype)
        k_hat = apply_rope(eff[0, ..., :d], cos_, sin_).float()      # [H,M,d]
        v_hat = eff[0, ..., d:].float()

        for h in range(H):
            ev = (~valid[h]).nonzero(as_tuple=True)[0]
            if ev.numel() < 64:
                continue
            k_ev, v_ev = kall[h, ev], vall[h, ev]
            # 位置局部质心（与 centroid.py 的 post 模式同构：不旋转、直接平均）
            nb = min(M, 64)
            blk = torch.linspace(0, len(ev), nb + 1).long()
            k_c = torch.stack([k_ev[blk[i]:blk[i + 1]].mean(0) for i in range(nb)])
            v_c = torch.stack([v_ev[blk[i]:blk[i + 1]].mean(0) for i in range(nb)])
            cnt = torch.tensor([float(blk[i + 1] - blk[i]) for i in range(nb)],
                               device=k_c.device)

            # ---- A. 可寻址性（读出帧）----
            nrm = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)  # noqa: E731
            a_learn = (nrm(k_ev) @ nrm(k_hat[h]).T).max(-1).values.mean()
            a_cent = (nrm(k_ev) @ nrm(k_c).T).max(-1).values.mean()

            for g in range(G):
                a = _Q[l][0].view(H, G, T, d)[h, g, -1].float() / (d ** 0.5)
                s_ev = a @ k_ev.T                                    # 真实被驱逐打分
                w_ev = torch.softmax(s_ev, -1)
                o_E = w_ev @ v_ev                                    # oracle 的 o_E
                # ---- B. 打分保持：真实 top-10% 的质量在记忆里占多少 ----
                k_top = max(1, len(s_ev) // 10)
                top = s_ev.topk(k_top).indices
                # learned：槽内 softmax（与读出一致）
                r_l = torch.softmax(a @ k_hat[h].T, -1)
                m_l = r_l @ v_hat[h]
                # 质心：带 log n 的共享式打分（与 centroid.py 一致）
                r_c = torch.softmax(a @ k_c.T + cnt.log(), -1)
                m_c = r_c @ v_c
                # ---- C. value 方向 ----
                cs = lambda x, y: float(torch.dot(x, y) / (x.norm() * y.norm() + 1e-9))  # noqa: E731
                rows.append((l, h, g, float(a_learn), float(a_cent),
                             cs(m_l, o_E), cs(m_c, o_E),
                             float(w_ev[top].sum())))
    A = np.array(rows)
    print("\n" + "=" * 96)
    print(f"取证：{ds_name} @ratio {RATIO}　样本 0　{len(A)} 个 (层,头,组)　"
          f"learned M={M}，质心 {min(M,64)} 块")
    print("**全部在读出帧里比较** —— learned 槽已 apply_rope 到位置质心，与推理一致")
    print("-" * 96)
    print(f"{'指标':<40}{'learned':>12}{'质心':>12}{'倍数':>10}")
    md = lambda c: float(np.median(A[:, c]))                          # noqa: E731
    print(f"{'A. 可寻址性 mean_i max_j cos(k_i, k̂_j)':<40}{md(3):>12.4f}{md(4):>12.4f}"
          f"{md(4)/max(md(3),1e-9):>10.2f}×")
    print(f"{'C. value 方向 cos(m(q), o_E(q))':<40}{md(5):>12.4f}{md(6):>12.4f}"
          f"{md(6)/max(abs(md(5)),1e-9):>10.2f}×")
    print(f"{'参考：真实 top-10% 被驱逐 token 占的质量':<40}{md(7):>12.4f}")
    print("=" * 96)
    print("判读：")
    print("  A 明显更低 ⇒ **encoder/槽/decoder 洗掉了 key 身份** ⇒ d_z=64 瓶颈值得测")
    print("  A 相当但 C 更低 ⇒ 寻址没坏、**value 重建/读出**坏了 ⇒ 改读出而非维度")
    print("  两个都相当却下游不涨 ⇒ 问题在注入方式或选择性，不在表示")


if __name__ == "__main__":
    main()
