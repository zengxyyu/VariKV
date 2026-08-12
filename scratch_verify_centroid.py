"""质心读出的验收检查。代数错了整轮容量实验就白跑，所以这些必须先过。

四项，从弱到强：

  1. **空记忆 ⇒ 与基线逐字相同。** 尚未吸收任何东西时 λ=1，输出必须与
     `RetainCache` 一模一样（不是"接近"，是 bit-identical）。
  2. **每个被驱逐 KV 单独成簇（W=1, n=1）⇒ 恢复满缓存注意力。** 这是最强的一项：
     此时 (k̄_j, v̄_j, n_j) = (k_j, v_j, 1)，代数应精确等于
     softmax over (保留 ∪ 被驱逐) = 未剪枝的注意力。**这一项同时验了
     flash 的 lse 语义、λ 混合、以及 log n_j 的符号。**
  3. **λ 与实测缺失质量一致。** 独立算出 M = D_E/(D_R+D_E)，与读出用的 (1−λ) 对账。
  4. **计数项确实在起作用。** 把 log n_j 抽掉，(1−λ) 应塌掉约 e^{log n̄} 倍 ——
     量化"一个槽只算一票"这个旧 bug 的大小。

用法： .venv/bin/python scratch_verify_centroid.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip                     # noqa: E402
from attention.centroid import CentroidRetainCache       # noqa: E402
from data.load import load_dataset_all                   # noqa: E402
from data.wrapper import DataWrapper, get_query          # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
DATA = "scbench_kv"
RATIO = 0.1
CHUNK, WINDOW, LEVEL = 16000, 4096, "pair"


def build(m, dw, si, kv_type, **kw):
    m.kv_type = kv_type
    for k, v in kw.items():
        setattr(m, k, v)
    return dw.prefill_context(si, prefill_chunk=CHUNK, window_size=WINDOW,
                              chunk_ratio=RATIO, level=LEVEL)


@torch.no_grad()
def main():
    m = ModelKVzip(MODEL, kv_type="retain", gate_path_or_name="fastkvzip")
    ds = load_dataset_all(DATA, m.tokenizer)
    dw = DataWrapper(DATA, ds, m)
    q = m.apply_template(get_query("qa", list(ds[0]["question"])[0])).to(m.device)
    ok = True

    # ---------------------------------------------------------------- 1
    print("\n[1] 空记忆 ⇒ 与基线逐字相同", flush=True)
    kv = build(m, dw, 0, "retain")
    S = kv.key_cache[0].shape[2]
    ref = m.model(q, past_key_values=kv).logits[0].float().cpu()
    kv.slice(S)
    del kv
    torch.cuda.empty_cache()

    kvc = build(m, dw, 0, "centroid", varikv_K=109, varikv_rope_mode="post")
    Sc = kvc.key_cache[0].shape[2]
    b = kvc.budget()
    print(f"    吸收 {b['absorbed_kv']} 条 KV → 占用 {b['clusters_occupied']} 簇 "
          f"(K={b['K_per_head']}/head, W={kvc.W})；保留 {b['retained_kv']} 条 KV")
    print(f"    额外开销 = {b['scalars_centroid']/1e6:.2f}M / "
          f"{b['scalars_retained']/1e6:.1f}M scalars = {b['overhead_frac']*100:.3f}%")
    kvc.assert_causal(kvc.sink + kvc.ctx_len)

    kvc.centroid_mode = False          # 关掉 ⇒ 必须退回基线
    kvc.need_lse = False
    off = m.model(q, past_key_values=kvc).logits[0].float().cpu()
    kvc.slice(Sc)
    d0 = (off - ref).abs().max().item()
    print(f"    centroid_mode=False:  max|Δlogit| = {d0:.3e}  "
          f"{'✓ 逐字相同' if d0 == 0 else '✗'}")
    ok &= d0 == 0

    kvc.centroid_mode = True
    kvc.need_lse = True
    on = m.model(q, past_key_values=kvc).logits[0].float().cpu()
    kvc.slice(Sc)
    d1 = (on - ref).abs().max().item()
    print(f"    centroid_mode=True :  max|Δlogit| = {d1:.3e}  "
          f"{'✓ 修正确实改变了输出' if d1 > 1e-3 else '✗ 没有生效！'}")
    ok &= d1 > 1e-3

    # ---------------------------------------------------------------- 3 + 4
    print("\n[3] λ 与独立算出的缺失质量对账 / [4] log n_j 的作用", flush=True)
    _QCAP = {}
    from attention.kvcache import RetainCache
    _orig = RetainCache.prepare

    def _p(self, qq, kk, vv, l):
        _QCAP[l] = qq.detach().clone()
        return _orig(self, qq, kk, vv, l)

    RetainCache.prepare = _p
    _QCAP.clear()
    m.model(q, past_key_values=kvc).logits
    kvc.slice(Sc)
    RetainCache.prepare = _orig

    H, d = kvc.n_heads_kv, kvc.head_dim
    rows = []
    for l in (0, 13, 26):
        T = _QCAP[l].shape[2]
        # valid 只覆盖 context；缓存 = sink + context + query，前后都恒为保留
        valid = kvc._get_valid(l, kvc.key_cache[l].shape[2])
        while valid.dim() > 2:
            valid = valid.squeeze(0)
        kall = kvc.key_cache[l][0]
        Gq = _QCAP[l].shape[1] // H
        kbar, vbar, logn = kvc._summary(l, torch.float32)
        for h in range(H):
            a = _QCAP[l][0].view(H, Gq, T, d)[h, 0, -1].float() / (d ** 0.5)
            sc = a @ kall[h].float().T                             # [S]
            m0 = sc.max()
            ev = ~valid[h]
            # 真实（用全部被驱逐 KV，只取因果范围内 = 全部 context）
            DE = torch.exp(sc[ev] - m0).sum()
            DR = torch.exp(sc[~ev] - m0).sum()
            M_true = (DE / (DR + DE)).item()
            # 估计（质心 + log n）
            r = a @ kbar[h].T + logn[h]
            LE = torch.logsumexp(r, -1)
            LR = torch.logsumexp(sc[~ev], -1)
            M_est = torch.sigmoid(LE - LR).item()                  # = 1−λ
            # 抽掉 log n_j（旧 bug 的形态）
            r0 = a @ kbar[h].T + torch.where(torch.isinf(logn[h]), logn[h],
                                             torch.zeros_like(logn[h]))
            M_nocnt = torch.sigmoid(torch.logsumexp(r0, -1) - LR).item()
            rows.append((l, h, M_true, M_est, M_nocnt,
                         float(logn[h][logn[h] > -1e29].mean())))
    print(f"    {'层':>3}{'头':>3}{'真实 M':>10}{'质心估计':>10}{'去掉logn':>10}"
          f"{'估/真':>8}{'log n̄':>8}")
    for l, h, mt, me, mn, ln in rows:
        print(f"    {l:>3}{h:>3}{mt:>10.4f}{me:>10.4f}{mn:>10.4f}"
              f"{me/max(mt,1e-9):>8.2f}{ln:>8.2f}")
    import numpy as np
    A = np.array([(r[2], r[3], r[4]) for r in rows])
    print(f"    中位：真实 M={np.median(A[:,0]):.4f}  估计={np.median(A[:,1]):.4f} "
          f"(比值 {np.median(A[:,1]/A[:,0]):.2f})  去掉 logn={np.median(A[:,2]):.4f} "
          f"(比值 {np.median(A[:,2]/A[:,0]):.4f})")
    print("    ⇒ 去掉 log n_j 后缺失质量塌成 ~1/e^{log n̄}，这就是旧 KV-injection "
          "只能还回 0.09% 的机制")
    del kvc
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 2（最强）
    print("\n[2a] 纯代数单元测试：λ 混合 ≡ 完整 softmax（不需要模型）", flush=True)
    g = torch.Generator().manual_seed(0)
    e2 = 0.0
    for _ in range(20):
        nR, nE, dd = 137, 991, 128
        a = torch.randn(dd, generator=g, dtype=torch.float64)
        kR = torch.randn(nR, dd, generator=g, dtype=torch.float64)
        vR = torch.randn(nR, dd, generator=g, dtype=torch.float64)
        kE = torch.randn(nE, dd, generator=g, dtype=torch.float64)
        vE = torch.randn(nE, dd, generator=g, dtype=torch.float64)
        sR, sE = kR @ a, kE @ a
        o_full = torch.softmax(torch.cat([sR, sE]), 0) @ torch.cat([vR, vE])
        LR, LE = torch.logsumexp(sR, 0), torch.logsumexp(sE, 0)
        lam = torch.exp(LR - torch.logaddexp(LR, LE))
        mix = lam * (torch.softmax(sR, 0) @ vR) + (1 - lam) * (torch.softmax(sE, 0) @ vE)
        e2 = max(e2, (mix - o_full).abs().max().item())
    print(f"     max|Δ| = {e2:.2e}  {'✓' if e2 < 1e-12 else '✗'}")
    ok &= e2 < 1e-12

    print("\n[2b] flash 的 lse 在真实前向里索引正确吗（= log D_R）", flush=True)
    kv2 = build(m, dw, 0, "centroid", varikv_K=109, varikv_rope_mode="post")
    S2 = kv2.key_cache[0].shape[2]
    CAP = {}
    _mc = CentroidRetainCache.memory_correct

    def _cap(self, qs, l, oR, lse):
        out = _mc(self, qs, l, oR, lse)
        if l in (0, 13, 26):
            CAP[l] = (qs.detach().clone(), oR.detach().clone(),
                      lse.detach().clone(), out.detach().clone())
        return out

    CentroidRetainCache.memory_correct = _cap
    m.model(q, past_key_values=kv2).logits
    CentroidRetainCache.memory_correct = _mc
    # **不要在这里 slice**：slice 会把 query token 从缓存删掉，而 flash 的 lse 是
    # 含 query token 的。先前在这里回滚导致参考少算 53 个 query key，差 18.79 nats。
    NSEQ = {l: kv2.key_cache[l].shape[2] for l in CAP}

    H, d = kv2.n_heads_kv, kv2.head_dim
    e_lse, e_out = 0.0, 0.0
    for l, (qs, oR, lse, out) in CAP.items():
        B, HQ, T, _ = qs.shape
        G = HQ // H
        valid = kv2._get_valid(l, NSEQ[l])
        while valid.dim() > 2:
            valid = valid.squeeze(0)
        kall = kv2.key_cache[l][0].float()
        kbar, vbar, logn = kv2._summary(l, torch.float32)
        lse_v = lse.float().view(G, H, T)
        for h in range(H):
            for gg in range(G):
                a = qs[0].view(H, G, T, d)[h, gg, -1].float() / (d ** 0.5)
                sc = a @ kall[h].T
                ref_LR = torch.logsumexp(sc[valid[h]], -1)        # 只对保留集
                e_lse = max(e_lse, abs(float(lse_v[gg, h, -1] - ref_LR)))
        # 逐项重算 memory_correct 的输出，验 einsum/permute/索引
        qh = qs.view(B, H, G, T, d)[0].float() * (d ** -0.5)
        oh = oR.view(B, T, H, G, d)[0].float()
        r = torch.einsum("hgtd,hkd->hgtk", qh, kbar) + logn[:, None, None, :]
        LE = torch.logsumexp(r, -1)
        oE = torch.einsum("hgtk,hkd->hgtd", torch.softmax(r, -1), vbar)
        LR = lse_v.permute(1, 0, 2)
        lam = torch.exp(LR - torch.logaddexp(LR, LE)).unsqueeze(-1)
        ref = (lam * oh.permute(1, 2, 0, 3) + (1 - lam) * oE)
        ref = ref.permute(2, 0, 1, 3).reshape(B, T, HQ * d)
        e_out = max(e_out, (ref.to(out.dtype) - out).abs().max().item())
    print(f"     lse vs 手算 log D_R : max|Δ| = {e_lse:.3e}  "
          f"{'✓ 索引与语义都对' if e_lse < 5e-2 else '✗ 布局错了'}")
    print(f"     memory_correct 复算 : max|Δ| = {e_out:.3e}  "
          f"{'✓' if e_out < 1e-5 else '✗'}")
    ok &= e_lse < 5e-2 and e_out < 1e-5

    print("\n[2c] 把摘要换成**精确**的被驱逐集合（单层）⇒ 必须恢复满缓存注意力",
          flush=True)
    l0 = 13
    valid = kv2._get_valid(l0, NSEQ[l0])
    while valid.dim() > 2:
        valid = valid.squeeze(0)
    kall = kv2.key_cache[l0][0].float()
    vall = kv2.value_cache[l0][0].float()
    qs = CAP[l0][0]
    T = qs.shape[2]
    G = qs.shape[1] // H
    worst = 0.0
    for h in range(H):
        ev = ~valid[h]
        for gg in range(G):
            a = qs[0].view(H, G, T, d)[h, gg, -1].float() / (d ** 0.5)
            sc = a @ kall[h].T
            o_full = torch.softmax(sc, -1) @ vall[h]
            LR = torch.logsumexp(sc[~ev], -1)
            LE = torch.logsumexp(sc[ev], -1)
            oR_ = torch.softmax(sc[~ev], -1) @ vall[h][~ev]
            oE_ = torch.softmax(sc[ev], -1) @ vall[h][ev]
            lam = torch.exp(LR - torch.logaddexp(LR, LE))
            mix = lam * oR_ + (1 - lam) * oE_
            worst = max(worst, (mix - o_full).abs().max().item() /
                        max(o_full.abs().max().item(), 1e-9))
    print(f"     相对误差 max = {worst:.2e}  "
          f"{'✓ 精确摘要恢复满缓存' if worst < 1e-4 else '✗'}")
    ok &= worst < 1e-4
    kv2.slice(S2)
    del kv2
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("全部通过 ✓" if ok else "有检查未通过 ✗ —— 不要开跑")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
