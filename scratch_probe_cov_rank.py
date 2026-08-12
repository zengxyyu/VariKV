"""对角 Σ_kk 高估投影方差多少倍？以及 Σ_kk 需要多少秩才够。

背景：MGF 的二阶项是 ½·Var(aᵀδ)。P0-D 用**真实投影方差**（直接从投影值算）测出
r_MGF 中位 0.973，看着很好。但若实现时用 diag(Σ_kk) 近似，得到的是 Σ_j a_j²σ_j²，
**忽略 key 各维之间的相关**，交叉项本该抵消掉一大部分 ⇒ 高估 ⇒ exp 溢出。
实测后果：E3/E4 直接出 NaN（2026-08-12）。

所以"二阶高斯够用"这个结论只在**能拿到真实投影方差**时成立，而这要求存储的协方差
能重建任意 query 方向上的方差 ⇒ 必须低秩，不能只存对角。本脚本量化需要多少秩。

逐 (层, kv_head, 簇) × 真实 query 方向，比较：
    v_true = Var(aᵀδ)                     真值
    v_diag = Σ_j a_j² σ_j²                对角近似
    v_rk   = Σ_j a_j² d_j + ‖U_rᵀa‖²      diag + rank-r（r = 1,2,4,8,16）
"""
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
    W = 8192
    RANKS = [1, 2, 4, 8, 16]
    m = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M", kv_type="retain",
                   gate_path_or_name="fastkvzip")
    L, H = m.config.num_hidden_layers, m.config.num_key_value_heads
    d = getattr(m.config, "head_dim",
                m.config.hidden_size // m.config.num_attention_heads)
    Gq = m.config.num_attention_heads // H
    ds = load_dataset_all("scbench_kv", m.tokenizer)
    dw = DataWrapper("scbench_kv", ds, m)
    acc = {"diag": []}
    acc.update({f"r{r}": [] for r in RANKS})

    for si in range(2):
        q0 = m.apply_template(get_query("qa", list(ds[si]["question"])[0]))
        kv = dw.prefill_context(si, prefill_chunk=16000, window_size=4096,
                                chunk_ratio=0.1, level="pair")
        _QCAP.clear()
        m.model(q0.to(m.device), past_key_values=kv)
        S = kv.key_cache[0].shape[2]
        for l in range(0, L, 4):
            valid = get_valid(kv, l, S).to(kv.key_cache[l].device)
            kh_all = kv.key_cache[l][0]
            for h in range(H):
                ev = (~valid[h]).nonzero(as_tuple=True)[0]
                if ev.numel() < 64:
                    continue
                kh = kh_all[h, ev].float()
                blk = ev // W
                a = _QCAP[l][0].view(H, Gq, -1, d)[h, 0, -1].float() / (d ** 0.5)
                for b in blk.unique():
                    sel = blk == b
                    if int(sel.sum()) < 64:
                        continue
                    X = kh[sel]
                    dX = X - X.mean(0)
                    v_true = float(((dX @ a) ** 2).mean())
                    if v_true < 1e-12:
                        continue
                    sig = (dX * dX).mean(0)
                    acc["diag"].append(float((a * a * sig).sum()) / v_true)
                    C = (dX.T @ dX) / dX.shape[0]
                    evals, evecs = torch.linalg.eigh(C.double())
                    for r in RANKS:
                        U = (evecs[:, -r:] * evals[-r:].clamp_min(0).sqrt()).float()
                        resid = (C.diag() - (U * U).sum(-1)).clamp_min(0)
                        v = float((a * a * resid).sum() + ((U.T @ a) ** 2).sum())
                        acc[f"r{r}"].append(v / v_true)
        del kv
        torch.cuda.empty_cache()
        print(f"样本{si} 累计 {len(acc['diag'])} 簇", flush=True)

    print("\n" + "=" * 78)
    print("投影方差近似 / 真值   （1 = 精确；≫1 = 高估 ⇒ exp 溢出）")
    for k, v in acc.items():
        v = np.array(v)
        if v.size == 0:
            continue
        print(f"  {k:<6} n={v.size:<6d} 中位={np.median(v):10.2f}  "
              f"P10={np.quantile(v,.1):9.2f}  P90={np.quantile(v,.9):10.2f}")
    print("=" * 78)
    print("判读：diag 若高估几十倍 ⇒ 对角协方差不可用，Σ_kk 必须低秩；")
    print("      中位最先接近 1 的那个 r，就是 Σ_kk 需要的最小秩（决定预算表）。")


if __name__ == "__main__":
    main()
