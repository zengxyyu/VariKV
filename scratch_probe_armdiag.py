#!/usr/bin/env python3
"""四臂的**机制**诊断 —— 让 "affine≈full ⇒ 是校准" 从暗示变成证据。

下游分数只告诉你哪条臂赢，不告诉你它靠什么赢。这个探针补两个量，跑一遍预填就够，
不用等评测：

1. **头内秩保持** `flip_lh` —— 同一个 (层,kv头) 内 `s⁰` 与 `s'` 的 Kendall τ、以及
   逐对翻转比例。这是"是校准不是重排序"的**充分条件检查**，不是修辞。

   为什么必须查：`bias` 臂的 `Δs = α·σ_h·tanh(b_lh)` 在头内是常数，所以
   `s'_i − s'_j = s_i − s_j`，**头内秩 100% 保持，无需测量**。但 `affine` 臂是

       s' = s⁰ + α σ_h tanh(a_h z + b_h),   z = (s⁰−μ_h)/σ_h
       ds'/ds⁰ = 1 + α a_h sech²(a_h z + b_h)

   `sech² ∈ (0,1]`，所以只有 `a_h > −1/α` 才保证处处单调。若某些头学到
   `a_h < −1/α`，`affine` 就在头内做了**局部重排序**，那它赢了也不能算校准的证据。

2. **逐 (层,头) 预算重分配** `B_lh` —— 全局阈值化后每个 (层,头) 实际拿到多少保留位，
   基线 vs 本臂。校准假设预测这里应有清晰的、系统性的再分配；若 `B_lh` 几乎不动而
   分数动了，那增益就不在预算通道上，校准故事不成立。

判读矩阵（配合下游分数一起读）：

| 下游 | flip_lh | B_lh 变化 | 结论 |
|---|---|---|---|
| affine≈full | ≈0 | 大 | **纯跨层头预算再校准** —— 最干净的方法命题 |
| affine≈full | 大 | 小 | 名为 affine 实为头内重排序，校准故事**不成立** |
| bias≈full | 0（构造） | 大 | 更强：只需 112 个平移量 |
| kv≫其它 | — | — | `(K,V)` 确实携带 gate 没抓到的 token 级信息 |

**命名**：`kv` 臂默认 `replace=False`，即 `s' = s⁰ + Δs(K,V)`，仍然以 FastKVzip 为
先验。报表里要写 **KV-residual**，不要写 "KV-only scorer" —— 后者是 `--replace`
那一档（独立打分器），两者的方法身份完全不同。
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
_P = os.path.join(ROOT, "external/FastKVzip/prefill")
sys.path.insert(0, _P)
os.chdir(_P)

from attention.calib_scorer import CalibScorer                  # noqa: E402
from data import DataWrapper, load_dataset_all                  # noqa: E402
from model import ModelKVzip                                    # noqa: E402


def kendall_flip(s0, s1, n_pair=20000, gen=None):
    """逐对抽样估 Kendall τ 与翻转比例。头内条目数 ~15 万，全对是 1e10 对，必须抽样。"""
    n = len(s0)
    if n < 2:
        return float("nan"), float("nan")
    i = torch.randint(0, n, (n_pair,), generator=gen)
    j = torch.randint(0, n, (n_pair,), generator=gen)
    k = i != j
    i, j = i[k], j[k]
    d0, d1 = (s0[i] - s0[j]), (s1[i] - s1[j])
    ok = d0.abs() > 1e-12
    d0, d1 = d0[ok], d1[ok]
    conc = (torch.sign(d0) == torch.sign(d1)).float()
    return float(2 * conc.mean() - 1), float(1 - conc.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="臂的 .pt（CalibScorer 或 ControlMemory）")
    ap.add_argument("-d", "--data", default="scbench_kv")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--num", type=int, default=4)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--chunk", type=int, default=16000)
    ap.add_argument("--window", type=int, default=4096)
    a = ap.parse_args()

    m = ModelKVzip(a.model, "retain", "fastkvzip")
    ds = DataWrapper(a.data, load_dataset_all(a.data, m.tokenizer), m)
    # **构造方式必须与 `eval_chunk.py:127-145` 逐字一致**，否则诊断的是另一个模型。
    # ckpt 顶层是 {state, mode, slots, dim, d_kv, L, H, arch}，不是 {cfg, model}。
    sd = torch.load(a.ckpt, map_location="cpu")
    arch = sd.get("arch", "memory")
    assert arch != "memory", "本探针只诊断 CalibScorer 四臂；ControlMemory 请走别的路径"
    ctrl = CalibScorer(sd.get("d_kv", 128), sd["L"], sd["H"],
                       n_slots=sd.get("slots", 8), d_m=sd.get("dim", 128),
                       mode="memoryless", arch=arch)
    ctrl.load_state_dict(sd["state"])       # strict：形状不符要崩，不要静默半加载
    ctrl.eval()
    print(f"[arm] {a.ckpt}  arch={arch}  alpha={float(ctrl.alpha):.4f}  "
          f"params={ctrl.n_params()}")
    gen = torch.Generator().manual_seed(0)
    taus, flips, dB = [], [], []

    for si in range(min(a.num, len(ds))):
        kv = ds.prefill_context(si, prefill_chunk=a.chunk, window_size=a.window,
                                chunk_ratio=a.ratio, level="pair")
        s0 = torch.stack(kv.score, 0)[:, 0].float()             # [L,H,n]
        sink, N = kv.sink, kv.valid.shape[-1]
        s0 = s0[..., sink:sink + N]
        L, H, _ = s0.shape
        # 复现训练时的 Δs：逐层调 read/delta，与 LearnedControlRetainCache 同一条路径
        s1 = s0.clone()
        gsig = s0.std()
        for l in range(L):
            st = ctrl.init_state(l)
            r = ctrl.read(st, None)
            sh = s0[l].cpu()
            mu = sh.mean(-1, keepdim=True)
            sg = sh.std(-1, keepdim=True).clamp_min(1e-6)
            # `kv`/`k`/`v` 臂的 delta 要真实 KV 特征；`bias`/`affine`/`scalar` 不用。
            # 必须走 raw→feat 两件套（`feat` 只收拼好的 [k;v]），与
            # `learned_ctrlcache.py:98-99` 同一条路径。
            x = None
            if arch in ("kv", "k", "v"):
                k_ = kv.key_cache[l][0][:, sink:sink + N].float().cpu()
                v_ = kv.value_cache[l][0][:, sink:sink + N].float().cpu()
                with torch.no_grad():
                    x = ctrl.feat(ctrl.raw(k_, v_))
                del k_, v_
            with torch.no_grad():
                d = ctrl.delta(x, r, sh, stats=(mu, sg, gsig.cpu()))
            s1[l] = sh.to(s1.device) + d.to(s1.device)
            del x

        for l in range(L):
            for h in range(H):
                t, f = kendall_flip(s0[l, h].cpu(), s1[l, h].cpu(), gen=gen)
                taus.append(t); flips.append(f)
        # 预算：全局阈值化后每个 (l,h) 拿到多少位
        def budget(s):
            f = s.reshape(-1)
            k = max(int(f.numel() * a.ratio) - 1, 0)
            thr = f.sort(descending=True).values[k]
            return (s > thr).sum(-1).float()                    # [L,H]
        b0, b1 = budget(s0), budget(s1)
        dB.append(((b1 - b0) / b0.clamp_min(1)).cpu().numpy())
        print(f"  样本 {si}: τ 中位 {np.median(taus[-L*H:]):.5f}  "
              f"翻转比例中位 {np.median(flips[-L*H:]):.2e}  "
              f"|ΔB|/B 中位 {np.median(abs(dB[-1])):.4f}", flush=True)
        del kv
        torch.cuda.empty_cache()

    D = np.concatenate([x.reshape(-1) for x in dB])
    print(f"\n=== {os.path.basename(a.ckpt)}  arch={arch}  "
          f"{min(a.num,len(ds))} 篇 ===")
    print(f"头内秩保持  Kendall τ 中位 {np.median(taus):.6f}   最小 {np.min(taus):.6f}")
    print(f"            逐对翻转比例 中位 {np.median(flips):.3e}  "
          f"P90 {np.percentile(flips,90):.3e}  最大 {np.max(flips):.3e}")
    print(f"逐 (层,头) 预算  |ΔB|/B 中位 {np.median(abs(D)):.4f}  "
          f"P90 {np.percentile(abs(D),90):.4f}  最大 {abs(D).max():.4f}")
    print(f"            净增头数 {int((D>0.01).sum())} / 净减 {int((D<-0.01).sum())}"
          f" / 共 {D.size}")
    print("\n判读：τ≈1 且 |ΔB|/B 大 ⇒ 纯跨层头预算再校准（最干净的方法命题）；"
          "\n      τ 明显<1 ⇒ 名为校准实为头内重排序，校准故事不成立；"
          "\n      |ΔB|/B≈0 ⇒ 增益不走预算通道，另找机制")


if __name__ == "__main__":
    raise SystemExit(main())
