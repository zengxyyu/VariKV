"""二阶矩摘要的两个瓶颈：分母（质量）与分子（方向），以及簇要多细才够。

起因（2026-08-12）：估计器阶梯里 E1（只用 μ_k、丢掉 E[e^δ] 因子）给 −80%，
而 E3/E4（加上 ½Var，分母更准）给 +2000~3000%（灾难）。两者不可能同时"对"。

第一轮对账（W=8192，138 簇）已定：
    分母：E1 的对数误差中位 −2.57（低估 13×），E4 只有 +0.06 ⇒ **二阶项是对的、必需的**
    分子：即使用**完整** Σ_vk，重建的 value 方向仍偏真值范数的 0.438（P90 0.956）
⇒ 瓶颈是**分子的方向**。E1 之所以"看起来好"，是因为它把方向错一半的修正整体缩小了 13 倍。

机制：Var(aᵀk) ≈ 5 ⇒ 簇内权重跨 e^{±2σ} ≈ 90 倍 ⇒ 倾斜均值由 8192 个成员里的少数几个主导，
而簇级 (μ, Σ) 无从知道是哪几个。

本脚本扫簇宽 W，回答决定性的问题：**簇细到什么程度，二阶摘要才够用？**
那个 W 对应的簇数 × 每簇字节 = 二阶矩路线真正需要的预算。
"""
import collections
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip                    # noqa: E402
from attention.kvcache import RetainCache               # noqa: E402
from data.load import load_dataset_all                  # noqa: E402
from data.wrapper import DataWrapper, get_query          # noqa: E402

_QCAP = {}
_orig = RetainCache.prepare


def _p(self, q, k, v, l):
    _QCAP[l] = q.detach().clone()
    return _orig(self, q, k, v, l)


RetainCache.prepare = _p


def get_valid(kv, l, S):
    try:
        v = kv._get_valid(l)
    except TypeError:
        v = kv._get_valid(l, S)
    v = v.bool()
    while v.dim() > 2:
        v = v.squeeze(0)
    return v


@torch.no_grad()
def main():
    WS = [128, 512, 2048, 8192, 32768]
    LAYERS = (0, 9, 14, 21, 26, 27)
    m = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M", kv_type="retain",
                   gate_path_or_name="fastkvzip")
    H = m.config.num_key_value_heads
    d = getattr(m.config, "head_dim",
                m.config.hidden_size // m.config.num_attention_heads)
    Gq = m.config.num_attention_heads // H
    ds = load_dataset_all("scbench_kv", m.tokenizer)
    dw = DataWrapper("scbench_kv", ds, m)
    q0 = m.apply_template(get_query("qa", list(ds[0]["question"])[0]))
    kv = dw.prefill_context(0, prefill_chunk=16000, window_size=4096,
                            chunk_ratio=0.1, level="pair")
    _QCAP.clear()
    m.model(q0.to(m.device), past_key_values=kv)
    S = kv.key_cache[0].shape[2]

    by_W = collections.defaultdict(list)
    for l in LAYERS:
        valid = get_valid(kv, l, S).to(kv.key_cache[l].device)
        kh_all = kv.key_cache[l][0]
        vh_all = kv.value_cache[l][0]
        for h in range(H):
            ev = (~valid[h]).nonzero(as_tuple=True)[0]
            if ev.numel() < 64:
                continue
            kh = kh_all[h, ev].double()
            vh = vh_all[h, ev].double()
            a = _QCAP[l][0].view(H, Gq, -1, d)[h, 0, -1].double() / (d ** 0.5)
            s = kh @ a
            for W in WS:
                blk = ev // W
                for b in blk.unique()[:8]:
                    sel = blk == b
                    n = int(sel.sum())
                    if n < 32:
                        continue
                    x = s[sel]
                    K_, V_ = kh[sel], vh[sel]
                    mu = float(x.mean())
                    V = float(x.var(unbiased=False))
                    L_true = float(torch.logsumexp(x, 0) - np.log(n))
                    w = torch.softmax(x, 0)
                    v_tilt = w @ V_                       # 真实倾斜均值
                    mu_v = V_.mean(0)
                    dk = K_ - K_.mean(0)
                    dv = V_ - mu_v
                    Svk = (dv.T @ dk) / n
                    U, Sg, Vt = torch.linalg.svd(Svk, full_matrices=False)
                    r = min(4, Sg.numel())
                    v_r4 = mu_v + (U[:, :r] * Sg[:r]) @ (Vt[:r] @ a)
                    v_full = mu_v + Svk @ a
                    nrm = float(v_tilt.norm()) + 1e-12
                    by_W[W].append((n, V, abs(mu + V / 2 - L_true),
                                    abs(mu - L_true),
                                    float((mu_v - v_tilt).norm()) / nrm,
                                    float((v_r4 - v_tilt).norm()) / nrm,
                                    float((v_full - v_tilt).norm()) / nrm))

    print("\n" + "=" * 108)
    print("簇宽 W 扫描  ——  value 误差 = ‖估计 − 真实倾斜均值‖ / ‖真值‖")
    print(f"{'W':>7}{'簇数':>7}{'n/簇':>7}{'Var中位':>9}{'|E1分母误差|':>13}"
          f"{'|E4分母误差|':>13}{'μ_v':>8}{'+Σvk r4':>9}{'+Σvk全':>8}{'P90(全)':>9}")
    for W in WS:
        rs = by_W[W]
        if not rs:
            continue
        A = np.array(rs)
        print(f"{W:>7}{len(rs):>7}{np.median(A[:,0]):>7.0f}{np.median(A[:,1]):>9.3f}"
              f"{np.median(A[:,3]):>13.3f}{np.median(A[:,2]):>13.3f}"
              f"{np.median(A[:,4]):>8.3f}{np.median(A[:,5]):>9.3f}"
              f"{np.median(A[:,6]):>8.3f}{np.quantile(A[:,6],.9):>9.3f}")
    print("=" * 108)
    print("判读：value 误差降到 <0.1 需要多细的 W？那个 W 的簇数 × 每簇字节 = 二阶矩路线")
    print("      真正需要的预算。若所需 W 小到簇数逼近 token 数，则该路线在预算上不成立。")


if __name__ == "__main__":
    main()
