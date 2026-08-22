"""ProMeta 接进 FastKVzip 的驱逐通路。

**设计原则：只改写 `self.score`，不碰 `threshold` / `valid` / `prepare`。**
`RetainCache.prune_chunk` 的全部工作是「取 `self.score` 的本块切片 → 阈值化
→ 追加 `self.valid`」。所以 ProMeta 只要在阈值化**之前**把分数换成风险分，
就能原样复用整条已经验证过的驱逐机械（包括 `level="pair"` 的全局阈值、
`adakv-layer` 的 safeguard、varlen kernel 的 `prepare`）。

**`mix=0` 必须逐位等同基线** —— 直接 `return super().prune_chunk(...)`，
连一次浮点运算都不多做。这是构造性零点对照，与地板那条线的 `b_min=0`
是同一个手法（`_fbm00` 实测逐样本 Δ 全零）。**没有这条，任何 ProMeta
读数都无法排除「通路本身扰动了掩码」。**

**上下文摘要取自 `value_cache[pool_layer]`，不是 hidden states。**
理由是**零 harness 改动**：hidden 要么得开 `save_hidden`（28 层全存 33 GB），
要么得往 `attn.py` 加钩子。而 V 已经在 cache 里、**未经 RoPE**（K 是 post-RoPE，
用它做位置无关的摘要不合适）。代价是摘要比 hidden 弱一档，`pool_layer`
留成可配项以便消融。

**⚠ 与 RestoreKV 的边界（构造性，不是措辞）**：probe 不进 `key_cache`／
`value_cache`，最终保留集恒为 `C' ⊂ C_original`，ProMeta 花的是**算力**
不是**预算**。
"""
import os

import torch

from prometa.model import ProMetaPredictor
from prometa.pool import OnlineAttnPool


def _z(x, eps=1e-6):
    """逐 (层,头) z-score。**混合两个分数前必须做** —— 门控分与风险分的量纲
    完全不同（前者是 logit 尺度、后者是 softmax 概率的 log），不归一化就等于
    让 `mix` 这个旋钮同时改变权重和量纲。"""
    m = x.mean(-1, keepdim=True)
    s = x.std(-1, keepdim=True).clamp_min(eps)
    return (x - m) / s


@torch.no_grad()
def prometa_scores(net, pool, key_cache, lo, hi, beta, device):
    """→ R: [L, Hkv, hi-lo]，熵风险聚合后的保留分。

    **整个函数在 `no_grad` 下**：ProMeta 的推理路径永远不需要梯度，
    而 `entropic_risk` 走 numpy，带 grad 的张量会直接抛
    `Can't call numpy() on Tensor that requires grad`（自测抓到）。
    """
    from prometa.risk import entropic_risk
    z = pool.value()                                       # [K,dp]
    q = net.from_pooled(z).detach()                        # [M,L,Hkv,d]
    M, L, H, d = q.shape
    out = torch.empty(L, H, hi - lo, device=device, dtype=torch.float32)
    for l in range(L):
        K = key_cache[l][0][:, lo:hi, :].to(q.dtype)       # [H,n,d]
        U = torch.softmax(
            torch.einsum("mhd,hnd->mhn", q[:, l], K) / d ** 0.5, dim=-1)
        out[l] = torch.as_tensor(
            entropic_risk(U.detach().float().cpu().numpy(), beta), device=device)
    return out


def make_prometa_cache(base_cls):
    """工厂：给任意 `RetainCache` 子类套上 ProMeta。**不改上游文件。**"""

    class ProMetaCache(base_cls):
        def pm_init(self, net: ProMetaPredictor, *, beta=1.0, mix=1.0,
                    pool_layer=14, verbose=True):
            self.pm_net = net.eval()
            self.pm_beta = float(beta)
            self.pm_mix = float(mix)
            self.pm_pool_layer = int(pool_layer)
            self.pm_pool = None
            self.pm_score0 = None
            self.pm_verbose = verbose
            self.pm_nchunk = 0
            return self

        def _pm_update_pool(self):
            """用 `value_cache[pool_layer]` 的**新增部分**更新在线池化。"""
            l = self.pm_pool_layer
            V = self.value_cache[l][0]                     # [Hkv,N,d]
            N = V.shape[1]
            if self.pm_pool is None:
                self.pm_pool = OnlineAttnPool(
                    self.pm_net.pool_q, device=V.device)
            seen = self.pm_pool.n
            if N <= seen:
                return
            new = V[:, seen:, :].permute(1, 0, 2).reshape(N - seen, -1)
            self.pm_pool.update(self.pm_net.proj(new.to(self.pm_net.proj.weight.dtype)))

        def prune_chunk(self, ratio: float, evict_range=tuple, level: str = "pair"):
            # **mix=0 走原路，一次浮点都不多做** —— 构造性零点对照。
            if getattr(self, "pm_net", None) is None or self.pm_mix == 0.0:
                return super().prune_chunk(ratio, evict_range, level)

            lo, hi = evict_range
            with torch.no_grad():
                self._pm_update_pool()
                R = prometa_scores(self.pm_net, self.pm_pool, self.key_cache,
                                   lo, hi, self.pm_beta, self.key_cache[0].device)
                s0 = torch.stack(self.score, 0)[:, 0, :, lo:hi].float()  # [L,H,n]
                s = (1.0 - self.pm_mix) * _z(s0) + self.pm_mix * _z(R)
                # 写回本块切片（各 chunk 的 range 互不相交，安全），并留底备诊断
                if self.pm_score0 is None:
                    self.pm_score0 = {}
                self.pm_score0[(lo, hi)] = s0.cpu()
                for l in range(len(self.score)):
                    self.score[l][0, :, lo:hi] = s[l].to(self.score[l].dtype)

            out = super().prune_chunk(ratio, evict_range, level)

            if self.pm_verbose:
                # **每加一个 mode 必须同时加运行时日志**（本项目铁律）。
                # 缺了它，「ProMeta 到底动了没有」只能靠比分数间接推 ——
                # 地板那条线的 66 格空跑就是这么发生的。
                agree = None
                with torch.no_grad():
                    from prometa.risk import topb_mask
                    k = max(1, int(self.valid[..., -(hi - lo):].float().sum().item()
                                   / (s0.shape[0] * s0.shape[1])))
                    a = topb_mask(s0.cpu().numpy(), k)
                    b = topb_mask(_z(R).cpu().numpy(), k)
                    agree = float((a & b).sum() / max((a | b).sum(), 1))
                print(f"[prometa] chunk lo={lo} n={hi-lo} beta={self.pm_beta} "
                      f"mix={self.pm_mix} pool_layer={self.pm_pool_layer} "
                      f"pooled_n={self.pm_pool.n} "
                      f"R(mean={R.mean():.4e} std={R.std():.4e}) "
                      f"**J(base,prometa)@k={k}={agree:.4f}** "
                      f"kept={self.valid.float().mean():.4f}", flush=True)
            self.pm_nchunk += 1
            return out

    ProMetaCache.__name__ = f"ProMeta{base_cls.__name__}"
    return ProMetaCache


def _selftest():
    """CPU 自测：只测新逻辑（z-score、mix 旋钮语义、risk 对拍）。
    `prune_chunk` 需要真 cache，放 GPU 冒烟（`scratch_prometa_smoke.py`）。"""
    # ① z-score：逐 (层,头) 独立、均值 0 方差 1
    x = torch.randn(3, 4, 50) * 7 + 3
    zx = _z(x)
    assert zx.mean(-1).abs().max() < 1e-5 and (zx.std(-1) - 1).abs().max() < 1e-3
    print("① _z 逐 (层,头) 归一化　PASS")

    # ② mix 旋钮两端语义
    a, b = torch.randn(2, 2, 20), torch.randn(2, 2, 20)
    for mix, ref in [(0.0, _z(a)), (1.0, _z(b))]:
        assert (((1 - mix) * _z(a) + mix * _z(b)) - ref).abs().max() < 1e-6, mix
    print("② mix 旋钮两端语义正确　PASS")

    # ③ **不做 z-score 的阴性对照**：量纲差 100× 时 mix=0.5 会被大尺度那侧支配。
    #    用秩相关衡量「混合结果更像谁」。
    big, small = torch.randn(200) * 100, torch.randn(200)
    def sp(u, v):
        ru = torch.argsort(torch.argsort(u)).float()
        rv = torch.argsort(torch.argsort(v)).float()
        ru -= ru.mean(); rv -= rv.mean()
        return float((ru * rv).sum() / (ru.norm() * rv.norm()))
    raw = 0.5 * big + 0.5 * small
    zed = 0.5 * _z(big[None, None]) [0, 0] + 0.5 * _z(small[None, None])[0, 0]
    assert sp(raw, big) > 0.99, sp(raw, big)
    assert abs(sp(zed, big) - sp(zed, small)) < 0.35, (sp(zed, big), sp(zed, small))
    print(f"③ 阴性对照：未归一化时 ρ(mix, big)={sp(raw,big):.4f}（被大尺度支配）；"
          f"归一化后 ρ 对 big/small = {sp(zed,big):.3f}/{sp(zed,small):.3f}　PASS")

    # ④ `prometa_scores` 与直接算 risk 对拍（用假 key_cache）
    from prometa.risk import entropic_risk
    from prometa.pool import OnlineAttnPool
    import numpy as np
    L, H, d, N, M = 2, 3, 8, 30, 5
    net = ProMetaPredictor(H * d, d, L, H, n_future=M, d_proj=8, n_pool=2, d_lat=4)
    kc = [torch.randn(1, H, N, d) for _ in range(L)]
    pool = OnlineAttnPool(net.pool_q)
    pool.update(net.proj(torch.randn(N, H * d)))
    R = prometa_scores(net, pool, kc, 5, 25, 1.3, torch.device("cpu"))
    assert R.shape == (L, H, 20), R.shape
    q = net.from_pooled(pool.value())
    Uref = torch.softmax(torch.einsum("mhd,hnd->mhn", q[:, 1], kc[1][0][:, 5:25]) / d ** 0.5, -1)
    Rref = entropic_risk(Uref.detach().numpy(), 1.3)
    assert np.abs(Rref - R[1].numpy()).max() < 1e-5, np.abs(Rref - R[1].numpy()).max()
    print(f"④ prometa_scores 与直接算 risk 对拍 max|差| = "
          f"{np.abs(Rref - R[1].numpy()).max():.2e}　PASS")
    print("\nprometa/cache.py 自测 4 条（CPU 部分）全过")


if __name__ == "__main__":
    _selftest()
