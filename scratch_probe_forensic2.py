"""修正版取证：记忆注入的修正，在功能上逼近真正的 Δo 吗？

**为什么要有 v2。** `scratch_probe_forensic.py` 测的是 `cos(m(q), o_E(q))`，**目标选错了**。
残差读出是 `o = o_R + g·m(q)`，而满注意力是

    o_full = λ·o_R + (1−λ)·o_E,   λ = D_R/(D_R+D_E)
  ⇒ o_full − o_R = (1−λ)(o_E − o_R) ≡ Δo

所以记忆该逼近的是 **Δo**，不是 `o_E`。两者方向可以完全不同：
`o_R=[10,0]`、`o_E=[8,2]` 时 `o_E` 大致向右，而 `Δo ∝ [-2,2]` 方向完全不同。
于是 `cos(m,o_E)≈0` 完全可以与 `cos(g·m,Δo)≈1` 并存 —— v1 的取证会把一个**完美的
残差修正**误判成噪声。（这个恒等式本仓库早就在用，见 `scratch_probe_damage.py`
与 CLAUDE.md 的 "exact local counterfactual identity"；v1 取证用错了目标。）

三个指标，全部经 `W_O` 投影（value 空间的范数跨 head/layer 不可比，CLAUDE.md §1.6）：

  D1 方向    cos( W_O·δ̂_h , W_O·Δo_h )                逐头，修正方向对不对
  D2 幅度    ‖W_O·δ̂_h‖ / ‖W_O·Δo_h‖                   逐头，量级对不对
  D3 层级    cos( Σ_h W_O δ̂_h , Σ_h W_O Δo_h ) 与相对误差
             ← 这才是残差流真正看到的东西。P0 实测跨头相消只留下 0.25，
               所以逐头对齐好 ≠ 层级对齐好。

δ̂ 取的是**真正被注入的量**（`memory_residual` 的返回值，已含 σ(gate)），
不是解码出的裸 m(q)。质心臂按 `centroid.py` 的代数算出等效的加法修正
`(1−λ̂)(ô_E − o_R)`，两者因此可比。

用法：CUDA_VISIBLE_DEVICES=k .venv/bin/python scratch_probe_forensic2.py <dataset>
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

MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
CKPT = "varikv/ckpt_kl/s2b_dist_k16.pt"
RATIO, CHUNK, WINDOW, LEVEL = 0.1, 16000, 4096, "pair"
LAYERS = (0, 6, 13, 20, 26)

_Q = {}
_orig = RetainCache.prepare


def _p(self, q, k, v, l):
    _Q[l] = q.detach().clone()
    return _orig(self, q, k, v, l)


@torch.no_grad()
def main():
    ds_name = sys.argv[1] if len(sys.argv) > 1 else "scbench_prefix_suffix"
    RetainCache.prepare = _p
    m = ModelKVzip(MODEL, kv_type="memory_retain", gate_path_or_name="fastkvzip")
    from varikv.config import Config
    from varikv.memory import DistributionalMemory
    ck = torch.load(ROOT / CKPT, map_location=m.device)
    cfg = Config(); cfg.memory.num_slots = ck["num_slots"]
    H = m.config.num_key_value_heads; L = m.config.num_hidden_layers
    d = getattr(m.config, "head_dim", m.config.hidden_size // m.config.num_attention_heads)
    mem = DistributionalMemory(2 * d, cfg.memory, mode=ck["mode"],
                               n_groups=L * H).to(m.device, dtype=torch.float32)
    mem.load_state_dict(ck["memory"])
    m.varikv_memory = mem; m.varikv_M = ck["num_slots"]; m.varikv_residual = True
    rot = getattr(m.model.model, "rotary_emb", None)
    m.varikv_inv_freq = rot.inv_freq.detach().clone()

    ds = load_dataset_all(ds_name, m.tokenizer)
    dw = DataWrapper(ds_name, ds, m)
    q_ids = m.apply_template(get_query("qa", list(ds[0]["question"])[0])).to(m.device)
    kv = dw.prefill_context(0, prefill_chunk=CHUNK, window_size=WINDOW,
                            chunk_ratio=RATIO, level=LEVEL)
    _Q.clear()
    m.model(q_ids, past_key_values=kv)
    S = kv.key_cache[0].shape[2]
    M = ck["num_slots"]
    cs = lambda a, b: float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))  # noqa: E731

    per_head, per_layer = [], []
    for l in LAYERS:
        WO = m.model.model.layers[l].self_attn.o_proj.weight.detach().float()
        v_ = kv._get_valid(l, S)
        while v_.dim() > 2:
            v_ = v_.squeeze(0)
        valid = v_.bool().to(m.device)
        kall = kv.key_cache[l][0].float(); vall = kv.value_cache[l][0].float()
        T = _Q[l].shape[2]; G = _Q[l].shape[1] // H
        # δ̂：**真正被注入的量**（含 σ(gate)），形状 [B,T,HQ*d]
        dhat = kv.memory_correct if False else kv.memory_residual(_Q[l], l)
        dhat = dhat.view(1, T, H, G, d)[0, -1].float()          # [H,G,d]，取最后一个 query
        agg_l = torch.zeros(WO.shape[0], device=m.device)
        agg_h = torch.zeros(WO.shape[0], device=m.device)
        agg_c = torch.zeros(WO.shape[0], device=m.device)
        for h in range(H):
            ev = (~valid[h]).nonzero(as_tuple=True)[0]
            if ev.numel() < 64:
                continue
            k_ev, v_ev = kall[h, ev], vall[h, ev]
            k_rt, v_rt = kall[h, valid[h]], vall[h, valid[h]]
            # 位置局部质心（与 centroid.py 的 post 模式同构）
            nb = min(M, 64)
            blk = torch.linspace(0, len(ev), nb + 1).long()
            k_c = torch.stack([k_ev[blk[i]:blk[i+1]].mean(0) for i in range(nb)])
            v_c = torch.stack([v_ev[blk[i]:blk[i+1]].mean(0) for i in range(nb)])
            cnt = torch.tensor([float(blk[i+1]-blk[i]) for i in range(nb)], device=k_c.device)
            for g in range(G):
                hq = h * G + g
                W = WO[:, hq*d:(hq+1)*d]
                a = _Q[l][0].view(H, G, T, d)[h, g, -1].float() / (d ** 0.5)
                sR, sE = a @ k_rt.T, a @ k_ev.T
                LR, LE = torch.logsumexp(sR, -1), torch.logsumexp(sE, -1)
                oR = torch.softmax(sR, -1) @ v_rt
                oE = torch.softmax(sE, -1) @ v_ev
                lam = torch.exp(LR - torch.logaddexp(LR, LE))
                d_true = (1 - lam) * (oE - oR)                  # ← 正确的 Δo
                # 质心的等效加法修正
                rc = a @ k_c.T + cnt.log()
                LEc = torch.logsumexp(rc, -1)
                oEc = torch.softmax(rc, -1) @ v_c
                lamc = torch.exp(LR - torch.logaddexp(LR, LEc))
                d_cent = (1 - lamc) * (oEc - oR)
                dl, dc, dh = W @ d_true, W @ d_cent, W @ dhat[h, g]
                per_head.append((l, hq,
                                 cs(dh, dl), float(dh.norm()/dl.norm().clamp_min(1e-12)),
                                 cs(dc, dl), float(dc.norm()/dl.norm().clamp_min(1e-12)),
                                 cs(dh, dc)))
                agg_l += dl; agg_h += dh; agg_c += dc
        per_layer.append((l, cs(agg_h, agg_l), float(agg_h.norm()/agg_l.norm().clamp_min(1e-12)),
                          cs(agg_c, agg_l), float(agg_c.norm()/agg_l.norm().clamp_min(1e-12))))

    A = np.array(per_head); B = np.array(per_layer)
    md = lambda c: float(np.median(A[:, c]))                     # noqa: E731
    print("\n" + "=" * 100)
    print(f"修正版取证：{ds_name} @ratio {RATIO}　样本 0　{len(A)} 个 (层,查询头)")
    print(f"**目标是 Δo = o_full − o_R = (1−λ)(o_E − o_R)，不是 o_E**；全部经 W_O 投影")
    print("-" * 100)
    print(f"{'指标（中位数）':<44}{'learned':>13}{'质心':>13}")
    print(f"{'D1 方向  cos(W_O δ̂, W_O Δo)':<44}{md(2):>13.4f}{md(4):>13.4f}")
    print(f"{'D2 幅度  ‖W_O δ̂‖ / ‖W_O Δo‖':<44}{md(3):>13.4f}{md(5):>13.4f}")
    print(f"{'两者之间 cos(W_O δ̂_learned, W_O δ̂_centroid)':<44}{md(6):>13.4f}")
    print("-" * 100)
    print(f"{'D3 层级（跨头相加后）':<20}{'cos(learned)':>14}{'幅度比':>10}"
          f"{'cos(质心)':>12}{'幅度比':>10}")
    for l, c1, r1, c2, r2 in B:
        print(f"{'  layer '+str(int(l)):<20}{c1:>14.4f}{r1:>10.4f}{c2:>12.4f}{r2:>10.4f}")
    print(f"{'  中位':<20}{np.median(B[:,1]):>14.4f}{np.median(B[:,2]):>10.4f}"
          f"{np.median(B[:,3]):>12.4f}{np.median(B[:,4]):>10.4f}")
    print("=" * 100)
    print("判读（预注册）：")
    print("  learned D1 明显 >0 且 Retr.KV > Prefix-Suffix ⇒ 它在做**功能正确**的残差修正，")
    print("     只是不迁移 ⇒ functional-alignment 正则有依据（不是 raw-KV 重建）")
    print("  learned D1 ≈0 而下游仍 +21.60 ⇒ 成功机制**不是**局部注意力修正，")
    print("     而是某种全局补偿 ⇒ **千万不要**加局部结构损失，会杀掉它")
    print("  质心 D1 明显 >0 而 learned ≈0 ⇒ 两者携带互补信息，hybrid 值得做")


if __name__ == "__main__":
    main()
