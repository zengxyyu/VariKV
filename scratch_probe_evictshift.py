"""记忆是否改变了「哪些 KV 被驱逐」？—— 2×2 分解留下的唯一存活假设的直接验证

**背景。** `scratch_probe_4arm.py`（80 条）测出：纯 steering（内容全程为空、一直注入）
+1.00 未分离；纯记忆（吸收但预填不注入）+1.00 未分离；而完整方法 +40.00 ★。
全部增益来自**交互**（+38.00 ★）。

存活的机制假设只有一个：**记忆在预填期的注入改变 hidden states → 门控分数 →
哪些 KV 被驱逐**。也就是它在改变压缩本身，而不是事后补回被删的信息。这与取证一致 ——
它的修正方向与局部缺口 `Δo` 正交（cos≈0），因为它做的不是局部修补。

**本探针直接量这件事**：同一条样本、同一个 ratio，分别用
    B  `kv_type="retain"`（全程无记忆）
    F  `kv_type="memory_retain"` + 完整注入
预填，然后逐 (层, kv 头) 比较两者的 `valid` 掩码。

指标：
    IoU        保留集合的 Jaccard 相似度
    换掉比例   B 保留而 F 丢弃的占 B 保留量的比例（symmetric difference 的一半）
    |ΔN|       保留数量的差（检查是否只是预算漂移而非集合重排）

判据（预注册）：
    IoU ≈ 1.00（换掉 <1%）  ⇒ 驱逐决策几乎没变 ⇒ **该假设被否**，交互来自别处
    IoU 明显 <1（换掉 >5%） ⇒ 记忆确实在改变压缩本身 ⇒ 假设成立，
                              且"VariKV 是残差记忆"这个定位需要改成"它在改驱逐"

注意：`level="pair"` 会**跨层跨头全局**分配预算，所以 F 的总保留量可能与 B 不同；
必须同时看 |ΔN|，否则会把预算漂移误读成集合重排。

用法：CUDA_VISIBLE_DEVICES=k .venv/bin/python scratch_probe_evictshift.py --n 4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip                     # noqa: E402
from data.load import load_dataset_all                   # noqa: E402
from data.wrapper import DataWrapper                     # noqa: E402


def valid_of(kv, L, H):
    """→ [L, H, ctx] 的 bool，只取 context 段（不含 sink 与 query 的 padding）。"""
    out = []
    for l in range(L):
        v = kv.valid[l]
        v = torch.as_tensor(v).bool()
        while v.dim() > 2:
            v = v.squeeze(0)
        out.append(v.cpu())
    return torch.stack(out)                              # [L,H,ctx]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="varikv/ckpt_kl/s2b_point_k16.pt")
    ap.add_argument("--gate_scale", type=float, default=0.5)
    ap.add_argument("--data", default="scbench_kv")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--ratio", type=float, default=0.1)
    a = ap.parse_args()

    m = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M", kv_type="memory_retain",
                   gate_path_or_name="fastkvzip")
    from varikv.config import Config
    from varikv.memory import DistributionalMemory
    ck = torch.load(ROOT / a.ckpt, map_location=m.device)
    cfg = Config(); cfg.memory.num_slots = ck["num_slots"]
    H = m.config.num_key_value_heads; L = m.config.num_hidden_layers
    d = getattr(m.config, "head_dim", m.config.hidden_size // m.config.num_attention_heads)
    mem = DistributionalMemory(2 * d, cfg.memory, mode=ck["mode"],
                               n_groups=L * H).to(m.device, dtype=torch.float32)
    mem.load_state_dict(ck["memory"]); mem.eval()
    m.varikv_memory = mem; m.varikv_M = ck["num_slots"]; m.varikv_residual = True
    m.varikv_gate_scale = a.gate_scale
    rot = getattr(m.model.model, "rotary_emb", None)
    m.varikv_inv_freq = rot.inv_freq.detach().clone()

    ds = load_dataset_all(a.data, m.tokenizer)
    dw = DataWrapper(a.data, ds, m)
    P = dict(prefill_chunk=16000, window_size=4096, chunk_ratio=a.ratio, level="pair")

    rows = []
    for i in range(a.n):
        _kt = m.kv_type
        m.kv_type = "retain"
        kvB = dw.prefill_context(i, **P)
        vB = valid_of(kvB, L, H)
        del kvB; torch.cuda.empty_cache()
        m.kv_type = _kt
        kvF = dw.prefill_context(i, **P)
        vF = valid_of(kvF, L, H)
        del kvF; torch.cuda.empty_cache()
        n = min(vB.shape[-1], vF.shape[-1])
        b, f = vB[..., :n], vF[..., :n]
        inter = (b & f).sum(-1).float()
        union = (b | f).sum(-1).float()
        iou = (inter / union.clamp_min(1)).mean().item()
        dropped = ((b & ~f).sum(-1).float() / b.sum(-1).clamp_min(1).float()).mean().item()
        added = ((~b & f).sum(-1).float() / b.sum(-1).clamp_min(1).float()).mean().item()
        dN = (f.sum(-1).float() - b.sum(-1).float())
        rows.append((iou, dropped, added, b.sum(-1).float().mean().item(),
                     dN.mean().item(), dN.abs().mean().item()))
        print(f"  样本{i}: IoU {iou:.4f}  B保留而F丢 {100*dropped:5.2f}%  "
              f"F新保留 {100*added:5.2f}%  B保留量 {rows[-1][3]:.0f}  "
              f"ΔN 均值 {rows[-1][4]:+.0f}（|ΔN| {rows[-1][5]:.0f}）", flush=True)

    A = np.array(rows)
    print("\n" + "=" * 82)
    print(f"驱逐决策是否被记忆改变　{len(A)} 条　{a.data} @ratio {a.ratio}")
    print("-" * 82)
    print(f"  保留集合 IoU（B vs F）        {A[:, 0].mean():.4f}")
    print(f"  B 保留而 F 丢弃              {100*A[:, 1].mean():.2f}%")
    print(f"  F 新保留（B 丢弃）            {100*A[:, 2].mean():.2f}%")
    print(f"  每 (层,头) 平均保留量         {A[:, 3].mean():.0f}")
    print(f"  保留量差 ΔN 均值 / |ΔN| 均值  {A[:, 4].mean():+.1f} / {A[:, 5].mean():.1f}")
    print("=" * 82)
    print("判读：换掉 <1% ⇒ 驱逐几乎没变，该假设被否；>5% ⇒ 记忆在改变压缩本身。")
    print("      同时看 |ΔN| —— level=pair 全局分配预算，别把预算漂移误读成集合重排。")


if __name__ == "__main__":
    main()
