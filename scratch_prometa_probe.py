#!/usr/bin/env python3
"""ProMeta 判决探针：读 `scratch_prometa_oracle.py` 存的 `U[M,L,H,N]`，
回答**三个预注册问题**。全部零 GPU。

**这是止损开关，不是方法。** 本仓库撤回 49–63 的共同模式是「先建框架、
后做判决实验、框架死掉」。所以在写任何 Student / probe 网络之前，先用
**真实未来**（scbench 同一 context 自带 5 个 query）问：

  A（**杀手锏**）**同样预算下，按「未来效用均值」排和按「尾部风险」排，
     选出来的保留集是不是同一个？** 若 Jaccard ≈ 1 ⇒ 「风险敏感」与
     「期望效用」是同一个干预，ProMeta 的核心命题在本工作点上**没有内容**。
     —— 这与 2026-08-22 用 `cos(Δb_floor, Δb_shrink)=+0.9869` 零 GPU 判否
     「地板 vs 向均匀收缩」是同一套手法。
  B  **未来需求是多模态的吗？** 若所有位置都被同一个未来最大化，
     「多个可能未来」就退化成一个。
  C  **均值排序是否已经等价于最大值排序？** Spearman ≈ 1 ⇒ 分布无额外信息。

**判据全部写死在下面，且每条都配阴性对照**（`--selftest`）。

    .venv/bin/python scratch_prometa_probe.py --selftest
    .venv/bin/python scratch_prometa_probe.py scratch_prometa_oracle_*.npz
"""
import glob
import os
import sys

import numpy as np

RHO = 0.1          # 模拟的保留比例（与主实验的极端压缩工作点一致）
ALPHA = 0.75       # CVaR 分位：取「最需要它的那 25% 未来」的均值
J_DEAD = 0.95      # 判据 A 的死亡线


def cvar_upper(U, alpha=ALPHA):
    """上尾 CVaR：沿未来轴取 top-(1-alpha) 的均值。U: [M, ...]"""
    M = U.shape[0]
    k = max(1, int(round((1.0 - alpha) * M)))
    top = np.sort(U, axis=0)[-k:]
    return top.mean(axis=0)


def topk_mask(score, k):
    """按 score 取 top-k（沿最后一维），返回 bool 掩码。并列按索引先后。"""
    idx = np.argsort(-score, axis=-1, kind="stable")[..., :k]
    m = np.zeros(score.shape, dtype=bool)
    np.put_along_axis(m, idx, True, axis=-1)
    return m


def jaccard(a, b):
    inter = (a & b).sum(-1)
    union = (a | b).sum(-1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def spearman(x, y):
    rx = np.argsort(np.argsort(x, axis=-1), axis=-1).astype(np.float64)
    ry = np.argsort(np.argsort(y, axis=-1), axis=-1).astype(np.float64)
    rx -= rx.mean(-1, keepdims=True)
    ry -= ry.mean(-1, keepdims=True)
    num = (rx * ry).sum(-1)
    den = np.sqrt((rx ** 2).sum(-1) * (ry ** 2).sum(-1))
    return np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)


def analyse(U, label=""):
    """U: [M, L, H, N] → 判词由数字生成。"""
    M, L, H, N = U.shape
    k = max(1, int(round(RHO * N)))
    U = U.astype(np.float64)
    mean = U.mean(0)                       # [L,H,N]
    cv = cvar_upper(U)                     # [L,H,N]
    mx = U.max(0)

    jm_cv = jaccard(topk_mask(mean, k), topk_mask(cv, k))
    jm_mx = jaccard(topk_mask(mean, k), topk_mask(mx, k))
    sp = spearman(mean.reshape(-1, N), mx.reshape(-1, N))

    # B：多模态 —— 每个位置由哪个未来最大化；以及集中度
    arg = U.argmax(0)                      # [L,H,N]
    share = np.stack([(arg == m).mean() for m in range(M)])
    conc = (U.max(0) / np.maximum(U.mean(0) * M, 1e-30))   # 1/M=均匀, 1=单峰

    print(f"\n### {label}  U{U.shape}  ρ={RHO} ⇒ 每头保留 {k}/{N}\n")
    print("| 量 | 均值 | 中位 | min | max |")
    print("|---|---|---|---|---|")
    for nm, v in [("**A. Jaccard(mean, CVaR)**", jm_cv),
                  ("A'. Jaccard(mean, max)", jm_mx),
                  ("**C. Spearman(mean, max)**", sp.reshape(L, H))]:
        v = np.asarray(v).ravel()
        print(f"| {nm} | **{v.mean():.4f}** | {np.median(v):.4f} | "
              f"{v.min():.4f} | {v.max():.4f} |")
    print(f"\n**B. 各未来当「最需要者」的位置占比**："
          + " ・ ".join(f"m{i}={s:.3f}" for i, s in enumerate(share))
          + f"（均匀应为 {1/M:.3f}）")
    print(f"**B'. 集中度 `max/(M·mean)`**：均值 {conc.mean():.4f}"
          f"（{1/M:.3f}=完全均匀，1.0=单一未来独占）")

    print()
    if jm_cv.mean() >= J_DEAD:
        print(f"⇒ **判据 A 判否：Jaccard(mean, CVaR) = {jm_cv.mean():.4f} ≥ {J_DEAD}** "
              "⇒ 风险敏感与期望效用在本工作点上**选出同一个保留集**，"
              "ProMeta 的核心命题没有内容。**停止，不要写 Student。**")
    else:
        print(f"⇒ **判据 A 通过：Jaccard(mean, CVaR) = {jm_cv.mean():.4f} < {J_DEAD}** "
              "⇒ 两条规则确实选出不同的保留集。"
              "**⚠ 这只说明「不同」，不说明「更好」——** 下一步必须做"
              "下游对照（Oracle-Mean vs Oracle-CVaR 的真实任务分数），"
              "否则就是把「测量成立」当「处方成立」（第⑦类错）。")
    if share.max() > 0.9:
        print(f"⇒ **判据 B 判否：单个未来占 {share.max():.3f} 的位置** "
            "⇒ 未来需求不是多模态的，「多个可能未来」退化成一个。")
    return dict(jm_cv=float(jm_cv.mean()), jm_mx=float(jm_mx.mean()),
                sp=float(np.asarray(sp).mean()), share_max=float(share.max()))


def selftest():
    rng = np.random.default_rng(0)
    M, L, H, N = 5, 3, 4, 512
    print("阴性/阳性对照（判据本身也要能拒）：\n")
    # ① 所有未来完全相同 ⇒ mean 与 CVaR 必须选出同一个集合
    base = rng.random((1, L, H, N))
    same = np.repeat(base, M, axis=0)
    r = analyse(same, "① 五个未来完全相同（应 Jaccard=1、判否）")
    assert r["jm_cv"] > 0.999, r
    # ② 未来彼此独立 ⇒ 必须分得开
    ind = rng.random((M, L, H, N))
    r = analyse(ind, "② 五个未来彼此独立（应 Jaccard 明显 <1）")
    assert r["jm_cv"] < 0.9, r
    # ③ 单一未来独占 ⇒ 判据 B 必须判否
    sp1 = rng.random((M, L, H, N)) * 0.01
    sp1[0] += 1.0
    r = analyse(sp1, "③ 只有 m0 主导（判据 B 应判否）")
    assert r["share_max"] > 0.9, r
    # ④ topk_mask 的并列与计数
    s = np.zeros((2, 7)); s[0, [1, 3]] = 1.0
    m = topk_mask(s, 2)
    assert m.sum(-1).tolist() == [2, 2], m
    assert m[0].tolist() == [False, True, False, True, False, False, False], m[0]
    print("\n④ topk_mask 计数与并列                PASS")
    print("\nALL PASS")


def main():
    if "--selftest" in sys.argv:
        selftest(); return
    files = [a for a in sys.argv[1:] if a.endswith(".npz")] or \
        sorted(glob.glob("scratch_prometa_oracle_*.npz"))
    if not files:
        print("没有 npz —— 先跑 scratch_prometa_oracle.py。**这不是通过，是没数据。**")
        return
    Us = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        Us.append(d["U"])
        print(f"{os.path.basename(f)}: U{d['U'].shape} n_prefix={d['n_prefix']}")
    n = min(u.shape[-1] for u in Us)
    U = np.concatenate([u[..., :n] for u in Us], axis=1)   # 沿层轴拼样本
    analyse(U, f"合并 {len(files)} 个样本")


if __name__ == "__main__":
    main()
