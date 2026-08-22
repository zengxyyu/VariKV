"""ProMeta 接进 FastKVzip 的驱逐通路。

**设计原则：只改写 `self.score`，不碰 `threshold` / `valid` / `prepare`。**
`RetainCache.prune_chunk` 的全部工作是「取 `self.score` 的本块切片 → 阈值化
→ 追加 `self.valid`」。所以 ProMeta 只要在阈值化**之前**把分数换成风险分，
就能原样复用整条已经验证过的驱逐机械（包括 `level="pair"` 的全局阈值、
`adakv-layer` 的 safeguard、varlen kernel 的 `prepare`）。

**`gamma=0` 必须逐位等同基线** —— 直接 `return super().prune_chunk(...)`，
连一次浮点运算都不多做。这是构造性零点对照，与地板那条线的 `b_min=0`
是同一个手法（`_fbm00` 实测逐样本 Δ 全零）。**没有这条，任何 ProMeta
读数都无法排除「通路本身扰动了掩码」。**

────────────────────────────────────────────────────────────────────────
**⚠ 撤回（2026-08-22，写完当天复查查出，未上过 GPU）：首版的分数混合式**

    s = (1 − mix)·_z(s0) + mix·_z(R)            ← **错**

`_z(s0)` 把**每个 (层,头) 都拉成零均值单位方差**。而 `level="pair"` 是**跨全部
层头的单一全局阈值** —— 头能拿到多少配额，正是由 `s0` 的**逐头均值与尺度**
决定的。把它归一化掉，等于强行让所有头的分数分布重合，也就是**把配额向均匀
收缩**。本仓库已经离线判过：在 1100 条真实 `b⁰` 上，「向均匀收缩」与「反饿死
地板」的方向余弦是 **+0.9869** —— 是同一个干预，而地板在 Retr.KV@0.1 上值
**+33.60★**。⇒ 首版一开机就会把一个已知的大效应混进 ProMeta 的读数里，
无论 ProMeta 本身有没有信号都会「成功」。这正是第③类错（代理量当判据）与
第⑦类错（测量成立当处方成立）的组合。

**改成保留逐头位置与尺度的两种形式**（都以头自己的分数散布 `σ_h(s0)` 为单位）：

    resid  : s = s0 + γ · σ_h(s0) · z_h(R)          γ=0 ⇒ 逐位等同基线
    replace: s = μ_h(s0) + σ_h(s0) · z_h(R)         只换形状、不动头的位置/尺度

`resid` 与本仓库 `calib_scorer` 的 `s' = s⁰ + α·σ_h·tanh(φ)` 是**同一个参数化**
（§四之五 的等价定理正是在这个族上证的：保序逐头单调形变 ≡ 逐头配额重分配），
所以 ProMeta 的决策内容仍然落在那条定理覆盖的范围内，可以直接复用它的解释。
⚠ 两式都**不禁止**配额移动 —— 头内加零均值扰动会改变该头有多少条目越过全局
阈值。被禁止的只是「用归一化本身**系统性地**把配额推向均匀」。
────────────────────────────────────────────────────────────────────────

**上下文摘要取自 `value_cache[pool_layer]`，不是 hidden states。**
理由是**零 harness 改动**：hidden 要么得开 `save_hidden`（28 层全存 33 GB），
要么得往 `attn.py` 加钩子。而 V 已经在 cache 里、**未经 RoPE**（K 是 post-RoPE，
用它做位置无关的摘要不合适）。代价是摘要比 hidden 弱一档，`pool_layer`
留成可配项以便消融。

**⚠ 与 RestoreKV 的边界（构造性，不是措辞）**：probe 不进 `key_cache`／
`value_cache`，最终保留集恒为 `C' ⊂ C_original`，ProMeta 花的是**算力**
不是**预算**。
"""
import os as _os
import sys as _sys

import torch

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from prometa.model import ProMetaPredictor
from prometa.pool import OnlineAttnPool
from prometa.risk import entropic_risk_torch

COMBINE = ("resid", "replace")


def _z(x, eps=1e-6):
    """逐 (层,头) z-score（沿最后一维）。返回 `(z, mu, sigma)`。

    **只对 `R` 用。** 对 `s0` 用会摧毁逐头配额 —— 见模块 docstring 的撤回段。
    """
    m = x.mean(-1, keepdim=True)
    s = x.std(-1, keepdim=True).clamp_min(eps)
    return (x - m) / s, m, s


def combine_scores(s0, R, gamma, mode="resid"):
    """把风险分 `R` 并进门控分 `s0`，**保留 `s0` 的逐头位置与尺度**。

    `s0`, `R`: [L, H, n]。返回 [L, H, n]。
    `mode="resid"` 且 `gamma == 0` 时返回 `s0` 本身（调用方另有短路，这里也保证）。
    """
    assert mode in COMBINE, mode
    assert s0.shape == R.shape, (s0.shape, R.shape)
    zR, _, _ = _z(R)
    _, mu0, sd0 = _z(s0)
    if mode == "replace":
        return mu0 + sd0 * zR
    if gamma == 0.0:
        return s0
    return s0 + gamma * sd0 * zR


@torch.no_grad()
def prometa_scores(net, pool, key_cache, lo, hi, beta):
    """→ R: [L, Hkv, hi-lo]，熵风险聚合后的保留分。**全程留在 GPU。**

    **整个函数在 `no_grad` 下**：ProMeta 的推理路径永远不需要梯度。
    """
    z = pool.value()                                       # [K,dp]
    q = net.from_pooled(z).detach()                        # [M,L,Hkv,d]
    M, L, H, d = q.shape
    dev = key_cache[0].device
    out = torch.empty(L, H, hi - lo, device=dev, dtype=torch.float32)
    for l in range(L):
        K = key_cache[l][0][:, lo:hi, :].to(q.dtype)       # [H,n,d]
        U = torch.softmax(
            torch.einsum("mhd,hnd->mhn", q[:, l], K) / d ** 0.5, dim=-1)
        out[l] = entropic_risk_torch(U, beta).to(out.dtype)
    return out


def make_prometa_cache(base_cls):
    """工厂：给任意 `RetainCache` 子类套上 ProMeta。**不改上游文件。**"""

    class ProMetaCache(base_cls):
        def pm_init(self, net, *, beta=1.0, gamma=1.0,
                    combine="resid", pool_layer=14, verbose=True, oracle=None):
            """`net` 是 Student；`oracle` 是 `{(lo,hi): U[M,L,Hkv,n]}` 的**上界臂**。

            **Oracle 臂用未来查询算 `U` ⇒ 按定义泄漏**，它只能当任何 Student 的
            上界报，绝不能当方法。二者互斥（传了 oracle 就不看 net）。
            """
            assert (net is None) != (oracle is None), "net 与 oracle 恰给一个"
            self.pm_oracle = oracle
            self.pm_net = net.eval() if net is not None else None
            self.pm_beta = float(beta)
            self.pm_gamma = float(gamma)
            assert combine in COMBINE, combine
            self.pm_combine = combine
            self.pm_pool_layer = int(pool_layer)
            self.pm_pool = None
            self.pm_verbose = verbose
            self.pm_nchunk = 0
            self.pm_stats = []
            return self

        def _pm_noop(self):
            """`resid` + `gamma==0` 是**构造性零点**：短路回基线，一次浮点都不多做。
            `replace` 没有零点（它按定义就换掉了形状），不允许短路。"""
            if (getattr(self, "pm_net", None) is None
                    and getattr(self, "pm_oracle", None) is None):
                return True
            return self.pm_combine == "resid" and self.pm_gamma == 0.0

        def _pm_R(self, lo, hi):
            """本块的风险分 `R: [L,Hkv,hi-lo]`。Oracle 臂与 Student 臂唯一的分岔点。"""
            if self.pm_oracle is not None:
                U = self.pm_oracle.get((lo, hi))
                assert U is not None, \
                    f"oracle 表里没有 chunk ({lo},{hi})；表里有 {sorted(self.pm_oracle)[:4]}…"
                return entropic_risk_torch(U.float().to(self.key_cache[0].device),
                                           self.pm_beta)
            self._pm_update_pool()
            return prometa_scores(self.pm_net, self.pm_pool, self.key_cache,
                                  lo, hi, self.pm_beta)

        def _pm_update_pool(self):
            """用 `value_cache[pool_layer]` 的**新增部分**更新在线池化。"""
            l = self.pm_pool_layer
            V = self.value_cache[l][0]                     # [Hkv,N,d]
            N = V.shape[1]
            if self.pm_pool is None:
                self.pm_pool = OnlineAttnPool(self.pm_net.pool_q, device=V.device)
            seen = self.pm_pool.n
            if N <= seen:
                return
            new = V[:, seen:, :].permute(1, 0, 2).reshape(N - seen, -1)
            self.pm_pool.update(self.pm_net.proj(new.to(self.pm_net.proj.weight.dtype)))

        def prune_chunk(self, ratio: float, evict_range=tuple, level: str = "pair"):
            if self._pm_noop():
                return super().prune_chunk(ratio, evict_range, level)

            lo, hi = evict_range
            with torch.no_grad():
                R = self._pm_R(lo, hi)
                s0 = torch.stack(self.score, 0)[:, 0, :, lo:hi].float()  # [L,H,n]
                s = combine_scores(s0, R, self.pm_gamma, self.pm_combine)
                # 写回本块切片（各 chunk 的 evict_range 互不相交，安全）
                for l in range(len(self.score)):
                    self.score[l][0, :, lo:hi] = s[l].to(self.score[l].dtype)

            out = super().prune_chunk(ratio, evict_range, level)

            # **每加一个 mode 必须同时加运行时日志**（本项目铁律）。缺了它，
            # 「ProMeta 到底动了没有」只能靠比分数间接推 —— 地板那条线的
            # 66 格空跑就是这么发生的。
            with torch.no_grad():
                # `valid` 的维数在不同 cache 子类上不一定相同，**不猜形状**：
                # 取本块那一段、按 (层×头) 的个数折算出平均每头保留数。
                v = self.valid[..., -(hi - lo):]
                n_head = s0.shape[0] * s0.shape[1]
                kept = float(v.float().sum().item())
                k = max(1, min(hi - lo, int(round(kept / n_head))))
                a = s0.topk(k, dim=-1).indices
                b = s.topk(k, dim=-1).indices
                ma = torch.zeros_like(s0, dtype=torch.bool).scatter_(-1, a, True)
                mb = torch.zeros_like(s0, dtype=torch.bool).scatter_(-1, b, True)
                agree = float((ma & mb).sum().item()) / max(float((ma | mb).sum().item()), 1.0)
                st = dict(lo=lo, n=hi - lo, k=k, J=agree,
                          R_mean=float(R.mean()), R_std=float(R.std()),
                          kept=float(self.valid.float().mean()))
                self.pm_stats.append(st)
            if self.pm_verbose:
                src = "ORACLE" if self.pm_oracle is not None else \
                    f"student(pool_layer={self.pm_pool_layer},n={self.pm_pool.n})"
                print(f"[prometa] chunk lo={lo} n={hi-lo} beta={self.pm_beta} "
                      f"gamma={self.pm_gamma} combine={self.pm_combine} src={src} "
                      f"R(mean={st['R_mean']:.4e} std={st['R_std']:.4e}) "
                      f"**J(base,prometa)@k={k}={agree:.4f}** "
                      f"kept={st['kept']:.4f}", flush=True)
            self.pm_nchunk += 1
            return out

    ProMetaCache.__name__ = f"ProMeta{base_cls.__name__}"
    return ProMetaCache


# ────────────────────────────────── 自测 ──────────────────────────────────
def _selftest():
    """CPU 自测：只测新逻辑。`prune_chunk` 需要真 cache，见 `scratch_prometa_smoke.py`。"""
    import numpy as np
    torch.manual_seed(0)

    # ① _z 逐 (层,头) 归一化
    x = torch.randn(3, 4, 50) * 7 + 3
    zx, mu, sd = _z(x)
    assert zx.mean(-1).abs().max() < 1e-5 and (zx.std(-1) - 1).abs().max() < 1e-3
    assert (mu + sd * zx - x).abs().max() < 1e-4, "分解必须可逆"
    print("① _z 逐 (层,头) 归一化且分解可逆　PASS")

    # ② gamma=0（resid）逐位等同 s0；replace 不依赖 gamma
    s0 = torch.randn(3, 4, 60) * 2 + 5
    R = torch.randn(3, 4, 60)
    assert (combine_scores(s0, R, 0.0, "resid") - s0).abs().max().item() == 0.0
    r1 = combine_scores(s0, R, 0.0, "replace")
    r2 = combine_scores(s0, R, 9.9, "replace")
    assert (r1 - r2).abs().max().item() == 0.0
    print("② resid@γ=0 逐位等同 s0（差 0.0）；replace 与 γ 无关　PASS")

    # ③ **两种形式都保留逐头位置与尺度**（这是修 bug 的核心不变量）
    for mode, g in [("resid", 0.7), ("replace", 1.0)]:
        s = combine_scores(s0, R, g, mode)
        if mode == "replace":
            assert (s.mean(-1) - s0.mean(-1)).abs().max() < 1e-4
            assert (s.std(-1) - s0.std(-1)).abs().max() < 1e-4
        else:
            assert (s.mean(-1) - s0.mean(-1)).abs().max() < 1e-4
    print("③ replace 逐头均值/标准差与 s0 逐位一致；resid 逐头均值不变　PASS")

    # ④ **阴性对照：首版 `_z(s0)` 混合确实摧毁逐头配额。**
    #    造一批逐头均值差异很大的 s0（真实门控分就是这样：跨头 A_h 差 621×），
    #    用一个全局 top-B 模拟 `level="pair"`，看每个头拿到多少配额。
    L, H, N = 6, 4, 400
    base = torch.randn(L, H, N)
    off = torch.linspace(-3, 3, L * H).view(L, H, 1)
    scale = torch.linspace(0.3, 3.0, L * H).view(L, H, 1)
    s0b = base * scale + off                    # 逐头位置/尺度差异很大
    Rb = torch.randn(L, H, N)
    B = int(0.1 * L * H * N)

    def quotas(s):
        flat = s.reshape(-1)
        thr = flat.topk(B).values[-1]
        return (s > thr).reshape(L * H, N).sum(-1).float()

    q0 = quotas(s0b)
    q_bad = quotas(0.5 * _z(s0b)[0] + 0.5 * _z(Rb)[0])           # 首版（错）
    q_new = quotas(combine_scores(s0b, Rb, 0.5, "resid"))        # 现版
    q_rep = quotas(combine_scores(s0b, Rb, 1.0, "replace"))
    cv = lambda q: float(q.std() / q.mean().clamp_min(1e-9))
    # 首版必须把配额压向均匀（离散度大幅下降）；现版必须保住量级
    assert cv(q_bad) < 0.3 * cv(q0), (cv(q0), cv(q_bad))
    assert cv(q_new) > 0.7 * cv(q0), (cv(q0), cv(q_new))
    assert cv(q_rep) > 0.7 * cv(q0), (cv(q0), cv(q_rep))
    # 饿死头（配额 0）的个数：首版会把它们全救活 —— 那正是地板干的事
    z0, zb, zn = int((q0 == 0).sum()), int((q_bad == 0).sum()), int((q_new == 0).sum())
    print(f"④ 阴性对照 配额离散度 CV：基线 {cv(q0):.3f} → 首版 `_z(s0)` 混合 "
          f"{cv(q_bad):.3f}（塌向均匀）、现版 resid {cv(q_new):.3f}、"
          f"replace {cv(q_rep):.3f}")
    print(f"   饿死头数：基线 {z0} → 首版 {zb}（≈地板效应）、现版 {zn}　PASS")

    # ⑤ γ 的单调性：γ 越大，保留集离基线越远（J 单调不增）
    def jac(a, b, k):
        ia, ib = a.topk(k, -1).indices, b.topk(k, -1).indices
        ma = torch.zeros_like(a, dtype=torch.bool).scatter_(-1, ia, True)
        mb = torch.zeros_like(b, dtype=torch.bool).scatter_(-1, ib, True)
        return float((ma & mb).sum()) / float((ma | mb).sum())
    js = [jac(s0b, combine_scores(s0b, Rb, g, "resid"), 40)
          for g in (0.0, 0.1, 0.3, 1.0, 3.0)]
    assert js[0] == 1.0, js
    assert all(js[i] >= js[i + 1] - 1e-9 for i in range(len(js) - 1)), js
    print(f"⑤ J(base, resid@γ) 随 γ 单调不增：{[f'{v:.3f}' for v in js]}　PASS")

    # ⑥ `prometa_scores` 与直接算 risk 对拍（假 key_cache）
    from prometa.risk import entropic_risk
    Lc, Hc, d, Nc, M = 2, 3, 8, 30, 5
    net = ProMetaPredictor(Hc * d, d, Lc, Hc, n_future=M, d_proj=8, n_pool=2, d_lat=4)
    kc = [torch.randn(1, Hc, Nc, d) for _ in range(Lc)]
    pool = OnlineAttnPool(net.pool_q)
    pool.update(net.proj(torch.randn(Nc, Hc * d)))
    Rr = prometa_scores(net, pool, kc, 5, 25, 1.3)
    assert Rr.shape == (Lc, Hc, 20), Rr.shape
    q = net.from_pooled(pool.value())
    Uref = torch.softmax(
        torch.einsum("mhd,hnd->mhn", q[:, 1], kc[1][0][:, 5:25]) / d ** 0.5, -1)
    Rref = entropic_risk(Uref.detach().numpy(), 1.3)
    e = float(np.abs(Rref - Rr[1].numpy()).max())
    assert e < 1e-4, e
    print(f"⑥ prometa_scores 与 numpy risk 对拍 max|差| = {e:.2e}　PASS")

    # ⑦ `_pm_noop` 的三种情形
    class Dummy:
        pass
    C = make_prometa_cache(Dummy)
    o = C.__new__(C)
    o.pm_net = None; o.pm_oracle = None
    assert C._pm_noop(o)
    o.pm_net, o.pm_combine, o.pm_gamma = net, "resid", 0.0
    assert C._pm_noop(o)
    o.pm_gamma = 0.5
    assert not C._pm_noop(o)
    o.pm_combine, o.pm_gamma = "replace", 0.0
    assert not C._pm_noop(o), "replace 没有零点，不许短路"
    o.pm_net, o.pm_oracle, o.pm_combine, o.pm_gamma = None, {}, "resid", 0.5
    assert not C._pm_noop(o), "oracle 臂也应当进分支"
    print("⑦ _pm_noop：未装网络/resid@γ=0 短路；resid@γ>0、replace、oracle 臂"
          "都不短路　PASS")

    # ⑧ oracle 臂的 `_pm_R` 与直接算 risk 对拍，且缺表时必须报错（不许静默跑成基线）
    o2 = C.__new__(C)
    o2.pm_net, o2.pm_beta = None, 1.7
    Uo = torch.rand(5, Lc, Hc, 12)
    o2.pm_oracle = {(3, 15): Uo}
    o2.key_cache = [torch.zeros(1, Hc, 20, d)]
    Rg = C._pm_R(o2, 3, 15)
    from prometa.risk import entropic_risk as _er
    e8 = float(np.abs(_er(Uo.numpy(), 1.7) - Rg.numpy()).max())
    assert e8 < 1e-4, e8
    try:
        C._pm_R(o2, 0, 12)
        raise SystemExit("缺表没报错 —— 会静默跑成别的东西")
    except AssertionError as ex:
        assert "oracle 表里没有" in str(ex), ex
    print(f"⑧ oracle 臂 _pm_R 对拍 max|差| = {e8:.2e}；缺 chunk 时硬报错　PASS")

    print("\nprometa/cache.py 自测 8 条（CPU 部分）全过")


if __name__ == "__main__":
    _selftest()
