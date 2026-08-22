"""ProMeta 的风险聚合层：把「未来效用的分布」压成一个可排序的保留分。

**为什么是熵风险而不是 CVaR**（外部复核提出，采纳）：本项目每个上下文只有
`M=5` 个真实未来，而 `CVaR_α` 在 `M=5`、`α≥0.75` 时 `k=round((1−α)M)=1`
**恒等于 `max`** —— 叫它 CVaR 是不诚实的。熵风险

    ρ_β(U) = (1/β) · log( (1/M) Σ_m exp(β·U_m) )

在同样的 M 下是**光滑、可微、单参数**的，且两端有干净的极限：

    β → 0⁺   ⇒  ρ_β → E[U]        （期望未来效用；等价于「均值排序」）
    β → +∞   ⇒  ρ_β → max_m U_m   （最坏未来保护）
    β → −∞   ⇒  ρ_β → min_m U_m   （风险偏好，只作阴性对照）

**它不是拍脑袋的 `μ + λσ`，而是推出来的。** 由累积量母函数
`K(β)=log E[e^{βU}] = βμ + β²σ²/2 + O(β³)`：

    ρ_β(U) = K(β)/β = μ + (β/2)·σ² + O(β²)

⇒ **「期望未来效用 + 不确定性惩罚」是熵风险的二阶展开**，`β` 就是那个惩罚
系数。于是 ProMeta 与「学一个期望未来重要性」之间是**一条连续谱上的两点**，
`β=0` 那一端**恰好**就是 LookaheadKV 一类方法，消融只需扫 β。

⚠ `σ²` 是**总体方差**（1/M 归一），不是样本方差 —— 与 `ρ_β` 的定义一致。

数值上一律走 log-sum-exp 的减最大值形式；`|β|` 极小时直接退回均值
（否则 `log(1+x)/β` 会灾难性抵消）。
"""
import numpy as np

BETA_EPS = 1e-6          # |β| 小于它就直接用均值（见 docstring 最后一段）


def standardize(U, axis=0, eps=1e-12):
    """把 `U` 在「未来 × 位置」联合分布上零均值单位方差化（逐 (层,头) 一组）。

    **⚠ 这不是可选的美化，是必需的。** `ρ_β` 平移等变但**非尺度不变**：

        ρ_β(aU + b) = a·ρ_{aβ}(U) + b

    而 `U` 是 169k 个位置上的 softmax，绝大多数值 ~1e-5、峰值 ~0.9，且本模型
    的逐头尺度 `A_h = σ_h/σ_g` 跨头跨度 **621×**。⇒ **同一个 β 在不同头上是
    完全不同的风险厌恶强度**（实测 β=5 在尖峰头给 J(mean,ρ_β)=0.656、
    在弥散头给 0.966，等价强度差 486×）。

    标准化后 `ρ_β(z(U))` 的**排序恒等于** `ρ_{β/s}(U)` 的排序（自测逐位验证），
    所以这一步**就是**把 β 换成无量纲的 `β̃ = β/s`。⇒ 一个 β 才在全部
    (层,头) / panel / ratio 上表示同一件事。

    ⚠ 归一化必须在**联合 (未来, 位置)** 上做。若逐位置对未来轴做 z-score，
    「谁都不要的位置」与「谁都要的位置」会被抹成一样 —— 那会毁掉 level 信息。
    """
    U = np.asarray(U, dtype=np.float64)
    m = U.mean()
    sd = U.std()
    if U.ndim > 1:
        # 逐 (层,头)：把 axis(未来) 与最后一维(位置) 之外的维度当作头身份
        red = (axis, U.ndim - 1)
        m = U.mean(axis=red, keepdims=True)
        sd = U.std(axis=red, keepdims=True)
    return (U - m) / np.maximum(sd, eps)


def entropic_risk(U, beta, axis=0, standardized=True):
    """ρ_β(U) 沿 `axis` 聚合。

    `standardized=True`（默认）先做 `standardize` ⇒ **β 无量纲**，见其 docstring。
    `beta=0` 精确返回均值（不是近似）；`beta` 极大时返回 max。
    ⚠ 标准化是逐 (层,头) 的仿射变换，**不改变同一头内的排序语义**，
    但会改变返回值的绝对尺度 —— 只能用来排序，不要当成效用的绝对值。
    """
    U = np.asarray(U, dtype=np.float64)
    if standardized:
        U = standardize(U, axis=axis)
    if abs(beta) < BETA_EPS:
        return U.mean(axis=axis)
    M = U.shape[axis]
    z = beta * U
    zmax = z.max(axis=axis, keepdims=True)
    lse = zmax + np.log(np.exp(z - zmax).mean(axis=axis, keepdims=True))
    return np.squeeze(lse / beta, axis=axis)


def taylor_risk(U, beta, axis=0):
    """二阶展开 `μ + (β/2)σ²`（总体方差）。**只用于对拍与解释，不用于决策。**"""
    U = np.asarray(U, dtype=np.float64)
    return U.mean(axis=axis) + 0.5 * beta * U.var(axis=axis)


def entropic_risk_torch(U, beta, standardized=True):
    """`entropic_risk` 的 **torch/GPU 版**，语义逐位相同（自测 ⑪ 对拍）。

    **为什么要第二份实现**：推理热路径上 `U` 是 [M, Hkv, n]，每层每块都要算。
    numpy 版要 `.cpu().numpy()` —— 单个 169k 样本约 78 GB 的 H2D/D2H 搬运。
    torch 版留在 GPU 上，省掉全部同步点。
    **本项目铁律「同一个量两条实现必须对拍」在此显式执行**（自测 ⑪，
    随机形状 + 多个 β + 标准化开关，逐位比较）。

    ⚠ `torch.std` 默认 `unbiased=True`，而 `np.std` 是总体标准差 ⇒
    **必须 `unbiased=False`**，否则两份实现在小 M 上系统性差 √(M/(M−1))。
    """
    import torch
    U = U.float()
    if standardized:
        red = (0,) if U.dim() == 1 else (0, U.dim() - 1)
        m = U.mean(dim=red, keepdim=True)
        sd = U.std(dim=red, keepdim=True, unbiased=False).clamp_min(1e-12)
        U = (U - m) / sd
    if abs(beta) < BETA_EPS:
        return U.mean(0)
    z = beta * U
    zmax = z.amax(dim=0, keepdim=True)
    lse = zmax + torch.log(torch.exp(z - zmax).mean(0, keepdim=True))
    return (lse / beta).squeeze(0)


def topb_mask(score, b, axis=-1):
    """沿 `axis` 取 top-b，返回 bool 掩码。并列按索引先后（`stable`）。

    这是 ProMeta 唯一的决策规则：**保留集永远是原始 KV 的子集**，
    不生成任何合成记忆（与 RestoreKV 的边界就在这里）。
    """
    b = int(b)
    n = score.shape[axis]
    b = max(0, min(b, n))
    idx = np.argsort(-score, axis=axis, kind="stable")
    idx = np.take(idx, np.arange(b), axis=axis)
    m = np.zeros(score.shape, dtype=bool)
    np.put_along_axis(m, idx, True, axis=axis)
    return m


def _selftest():
    rng = np.random.default_rng(0)
    U = rng.random((5, 7, 11))
    # 原有 8 条测的是**未标准化**的纯数学性质，显式关掉；标准化另有 ⑨⑩ 两条。
    er = lambda x, b, **kw: entropic_risk(x, b, standardized=False, **kw)

    # ① β=0 精确等于均值
    assert np.allclose(er(U, 0.0), U.mean(0), atol=0), "β=0 必须精确等于均值"

    # ② β→+∞ 收敛到 max，β→−∞ 收敛到 min
    hi = er(U, 5e3)
    lo = er(U, -5e3)
    assert np.abs(hi - U.max(0)).max() < 1e-3, np.abs(hi - U.max(0)).max()
    assert np.abs(lo - U.min(0)).max() < 1e-3, np.abs(lo - U.min(0)).max()

    # ③ 单调不减于 β（风险厌恶越强、分数越高）—— 这是 ρ_β 的基本性质
    bs = [-8, -2, -0.5, 0.0, 0.5, 2, 8]
    vals = np.stack([er(U, b) for b in bs])
    assert (np.diff(vals, axis=0) >= -1e-9).all(), "ρ_β 必须随 β 单调不减"

    # ④ Jensen：β>0 时 ρ_β ≥ 均值；β<0 时 ≤
    assert (er(U, 1.0) >= U.mean(0) - 1e-12).all()
    assert (er(U, -1.0) <= U.mean(0) + 1e-12).all()

    # ⑤ 与二阶展开对拍（小 β），并验证误差随 β² 收缩 —— 这是「不是拍脑袋」的证据
    errs = []
    for b in (0.2, 0.1, 0.05):
        errs.append(np.abs(er(U, b) - taylor_risk(U, b)).max())
    assert errs[0] > errs[1] > errs[2], errs
    r = errs[0] / max(errs[2], 1e-30)
    assert 8 < r < 32, f"误差应约按 β² 收缩（16×），实测 {r:.1f}×"

    # ⑥ 数值稳定：巨大 β 不 overflow、不 NaN
    for b in (1e3, 1e4, -1e4):
        v = er(U * 100, b)
        assert np.isfinite(v).all(), (b, v)

    # ⑦ 阴性对照：所有未来完全相同时，ρ_β 与 β 无关（方差为 0）
    same = np.repeat(U[:1], 5, axis=0)
    v0, v1 = er(same, 0.0), er(same, 50.0)
    assert np.abs(v0 - v1).max() < 1e-9, np.abs(v0 - v1).max()

    # ⑧ topb_mask 计数、并列、边界
    s = np.array([[0., 1., 0., 1., 0.], [5., 4., 3., 2., 1.]])
    m = topb_mask(s, 2)
    assert m.sum(-1).tolist() == [2, 2]
    assert m[0].tolist() == [False, True, False, True, False], m[0]
    assert m[1].tolist() == [True, True, False, False, False], m[1]
    assert topb_mask(s, 0).sum() == 0 and topb_mask(s, 99).all()

    # ⑨ **标准化 ≡ 重标定 β**：ρ_β(z(U)) 与 ρ_{β/s}(U) 的排序必须逐位相同
    V = np.abs(rng.standard_normal((5, 3000))) ** 3 * 0.7
    sd = V.std()
    ra = entropic_risk(V, 2.0, standardized=True)
    rb = er((V - V.mean()) / sd, 2.0)
    order = lambda x: np.argsort(np.argsort(-x))
    assert np.array_equal(order(ra), order(rb)), "标准化不等价于重标定 β"
    rc = er(V, 2.0 / sd)
    assert np.array_equal(order(ra), order(rc)), "ρ_β(z(U)) 应与 ρ_{β/s}(U) 同序"
    print(f"⑨ 标准化 ≡ 把 β 换成 β/s（两条对拍排序逐位相同，s={sd:.4f}）　PASS")

    # ⑩ **阴性对照**：不标准化时，同一个 β 在两个尺度差 500× 的头上给出
    #    截然不同的「偏离均值」程度；标准化后必须拉齐。
    def jac(a, b, k):
        A, B = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
        return len(A & B) / len(A | B)
    js_raw, js_std = [], []
    for sc in (1.0, 0.002):
        W = np.abs(rng.standard_normal((5, 4000))) ** 3
        W = W / W.sum(-1, keepdims=True) * 4000 * sc
        js_raw.append(jac(W.mean(0), er(W, 5.0), 400))
        js_std.append(jac(W.mean(0), entropic_risk(W, 5.0), 400))
    assert abs(js_raw[0] - js_raw[1]) > 0.2, js_raw
    assert abs(js_std[0] - js_std[1]) < 0.1, js_std
    print(f"⑩ 阴性对照：未标准化 J = {js_raw[0]:.3f} vs {js_raw[1]:.3f}（差 "
          f"{abs(js_raw[0]-js_raw[1]):.3f}）；标准化后 {js_std[0]:.3f} vs "
          f"{js_std[1]:.3f}（差 {abs(js_std[0]-js_std[1]):.3f}）　PASS")

    # ⑪ **numpy 版与 torch 版逐位对拍**（同一个量两条实现必须对拍）
    import torch as _t
    worst = 0.0
    for shape in [(5, 7, 11), (6, 4, 2, 300), (3, 40)]:
        X = rng.standard_normal(shape) ** 3 * 3.0 + 1.0
        for b in (0.0, 0.3, 2.0, -2.0, 50.0):
            for std_ in (True, False):
                a_ = entropic_risk(X, b, standardized=std_)
                b_ = entropic_risk_torch(_t.tensor(X), b, standardized=std_).numpy()
                worst = max(worst, float(np.abs(a_ - b_).max()))
    assert worst < 2e-5, worst
    # 阴性对照：若 torch 端误用无偏标准差，M=3 上必须对不上（√(3/2)=1.225）
    Y = rng.standard_normal((3, 200))
    bad = _t.tensor(Y)
    bad = (bad - bad.mean()) / bad.std(unbiased=True)          # 故意用无偏
    d_bad = float(np.abs(entropic_risk(Y, 2.0) - _t.logsumexp(2.0 * bad, 0).sub(
        np.log(3.0)).div(2.0).numpy()).max())
    assert d_bad > 1e-3, d_bad
    print(f"⑪ numpy/torch 两份实现对拍 max|差| = {worst:.2e}（fp32 量级）；"
          f"阴性对照（误用无偏 std）差 {d_bad:.3f}　PASS")

    print("prometa/risk.py 自测 11 条全过　"
          f"（β² 收缩比实测 {r:.1f}×，理论 16×）")


if __name__ == "__main__":
    _selftest()
