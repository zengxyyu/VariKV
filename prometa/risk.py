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


def entropic_risk(U, beta, axis=0):
    """ρ_β(U) 沿 `axis` 聚合。U 可以是 numpy 数组。

    `beta=0` 精确返回均值（不是近似）；`beta` 极大时返回 max —— 两者都由
    log-sum-exp 的稳定形式自然给出，不需要分支特判上限。
    """
    U = np.asarray(U, dtype=np.float64)
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

    # ① β=0 精确等于均值
    assert np.allclose(entropic_risk(U, 0.0), U.mean(0), atol=0), "β=0 必须精确等于均值"

    # ② β→+∞ 收敛到 max，β→−∞ 收敛到 min
    hi = entropic_risk(U, 5e3)
    lo = entropic_risk(U, -5e3)
    assert np.abs(hi - U.max(0)).max() < 1e-3, np.abs(hi - U.max(0)).max()
    assert np.abs(lo - U.min(0)).max() < 1e-3, np.abs(lo - U.min(0)).max()

    # ③ 单调不减于 β（风险厌恶越强、分数越高）—— 这是 ρ_β 的基本性质
    bs = [-8, -2, -0.5, 0.0, 0.5, 2, 8]
    vals = np.stack([entropic_risk(U, b) for b in bs])
    assert (np.diff(vals, axis=0) >= -1e-9).all(), "ρ_β 必须随 β 单调不减"

    # ④ Jensen：β>0 时 ρ_β ≥ 均值；β<0 时 ≤
    assert (entropic_risk(U, 1.0) >= U.mean(0) - 1e-12).all()
    assert (entropic_risk(U, -1.0) <= U.mean(0) + 1e-12).all()

    # ⑤ 与二阶展开对拍（小 β），并验证误差随 β² 收缩 —— 这是「不是拍脑袋」的证据
    errs = []
    for b in (0.2, 0.1, 0.05):
        errs.append(np.abs(entropic_risk(U, b) - taylor_risk(U, b)).max())
    assert errs[0] > errs[1] > errs[2], errs
    r = errs[0] / max(errs[2], 1e-30)
    assert 8 < r < 32, f"误差应约按 β² 收缩（16×），实测 {r:.1f}×"

    # ⑥ 数值稳定：巨大 β 不 overflow、不 NaN
    for b in (1e3, 1e4, -1e4):
        v = entropic_risk(U * 100, b)
        assert np.isfinite(v).all(), (b, v)

    # ⑦ 阴性对照：所有未来完全相同时，ρ_β 与 β 无关（方差为 0）
    same = np.repeat(U[:1], 5, axis=0)
    v0, v1 = entropic_risk(same, 0.0), entropic_risk(same, 50.0)
    assert np.abs(v0 - v1).max() < 1e-9, np.abs(v0 - v1).max()

    # ⑧ topb_mask 计数、并列、边界
    s = np.array([[0., 1., 0., 1., 0.], [5., 4., 3., 2., 1.]])
    m = topb_mask(s, 2)
    assert m.sum(-1).tolist() == [2, 2]
    assert m[0].tolist() == [False, True, False, True, False], m[0]
    assert m[1].tolist() == [True, True, False, False, False], m[1]
    assert topb_mask(s, 0).sum() == 0 and topb_mask(s, 99).all()

    print("prometa/risk.py 自测 8 条全过　"
          f"（β² 收缩比实测 {r:.1f}×，理论 16×）")


if __name__ == "__main__":
    _selftest()
