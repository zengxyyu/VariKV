#!/usr/bin/env python3
"""ProMeta 判决探针：读 `scratch_prometa_oracle.py` 存的 `U[M,L,H,N]`，
回答**三个预注册问题**。全部零 GPU。

**这是止损开关，不是方法。** 本仓库撤回 49–63 的共同模式是「先建框架、
后做判决实验、框架死掉」。所以在写任何 Student / probe 网络之前，先用
**真实未来**（scbench 同一 context 自带 5 个 query）问：

  A（**杀手锏**）同预算下，按「未来效用**均值**」排和按「**尾部**」排，
     选出来的保留集是不是同一个？Jaccard ≥ 0.95 ⇒ 两条规则是同一个干预，
     ProMeta 的核心命题在本工作点上**没有内容**。
     —— 与 2026-08-22 用 `cos(Δb_floor, Δb_shrink)=+0.9869` 零 GPU 判否
     「地板 vs 向均匀收缩」是同一套手法。
  B  **未来需求是多模态的吗？** 若所有位置都被同一个未来最大化，
     「多个可能未来」退化成一个。
  C  均值排序是否已等价于最大值排序？

**五条已修的规格问题（写下来免得再犯）：**

1. **M=5 时 `CVaR_α` 对 α≥0.75 恒等于 `max`**（`k=round((1−α)·5)=1`）。
   所以本探针**不谎称在测 CVaR**：主判据用 `max`，另报一个 `k=2` 的
   真·尾部均值（α=0.6）。有效 `k` 一律打印。
2. **判据 B/C 必须限制在「有争议的位置」上。** 绝大多数前缀位置对所有
   未来都 ≈0，那里的 `argmax` 是纯噪声，会把多模态度稀释成均匀。
   现在同时报「全体位置」与「争议池（按 max 取 top-3k）」两栏。
3. **预算不能只取一个。** ρ 扫 {0.02, 0.05, 0.1, 0.2}；结论若随 ρ 变号，
   就不是结论。
4. **面板会左右判据 A，而方向与直觉相反。** 位置分两类：
   被**单个**未来以强度 `u` 需要（`mean=u/M`、`max=u`），或被**全部** `M` 个
   未来以强度 `v` 需要（`mean=max=v`）。mean 把后者排前当且仅当 `v > u/M`，
   max 把前者排前当且仅当 `u > v` ⇒ **两条规则要分歧，必须「私有强需求」与
   「共享弱需求」同时存在**。若未来**互不相交**（只有第一类），
   `mean = max/M` 是单调变换 ⇒ **排序完全相同 ⇒ 判据 A 判否**。
   ⇒ `scbench_kv` 的 5 个未来是 5 个互不相交的 key 查找，**对判据 A 偏悲观**；
   `scbench_qa_eng` 是同一文档上的 5 个自然语言问题，共享背景 + 各自证据，
   **是分歧最可能出现的面板**。**两个都要跑，只在一个上得到结论都不算。**
   （⚠ 这一条我第一版写反了 —— 曾把 `scbench_kv` 标成「偏乐观」。
   合成夹具上「未来互不相交 ⇒ J≈1」把它纠正过来。）
5. **判据 A 只是必要条件。** 「选出的集合不同」≠「风险规则更好」。
   通过后必须做下游对照（Oracle-Mean vs Oracle-Tail 的真实任务分数），
   否则就是把「测量成立」当「处方成立」（第⑦类错）。

    .venv/bin/python scratch_prometa_probe.py --selftest
    .venv/bin/python scratch_prometa_probe.py scratch_prometa_oracle_*.npz
"""
import glob
import os
import re
import sys

import numpy as np

RHOS = (0.02, 0.05, 0.10, 0.20)
J_DEAD = 0.95
CONTEND = 3          # 争议池 = 按 max 取 top-(CONTEND·k)


def tail_mean(U, alpha):
    """上尾均值。返回 (值, 有效 k)。k==1 时它就是 max —— 调用方必须报出来。"""
    M = U.shape[0]
    k = max(1, int(round((1.0 - alpha) * M)))
    return np.sort(U, axis=0)[-k:].mean(axis=0), k


def topk_mask(score, k):
    idx = np.argsort(-score, axis=-1, kind="stable")[..., :k]
    m = np.zeros(score.shape, dtype=bool)
    np.put_along_axis(m, idx, True, axis=-1)
    return m


def jaccard(a, b):
    inter = (a & b).sum(-1)
    union = (a | b).sum(-1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def spearman(x, y):
    rx = np.argsort(np.argsort(x, -1), -1).astype(np.float64)
    ry = np.argsort(np.argsort(y, -1), -1).astype(np.float64)
    rx -= rx.mean(-1, keepdims=True); ry -= ry.mean(-1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(-1) * (ry ** 2).sum(-1))
    return np.where(den > 0, (rx * ry).sum(-1) / np.maximum(den, 1e-30), 0.0)


def analyse(U, label="", quiet=False):
    """U: [M,L,H,N] → dict。判词由数字生成。"""
    M, L, H, N = U.shape
    U = U.astype(np.float64)
    mean = U.mean(0)
    mx = U.max(0)
    t2, k2 = tail_mean(U, 0.60)          # M=5 ⇒ k=2，真·尾部均值
    _, k75 = tail_mean(U, 0.75)

    # **判据 A 的一个真实陷阱（自测⑦ 抓到，比我预想的更糟）**：
    # 若绝大多数位置对所有未来都 ≈0，而每头保留数 `k` 超过「有实际质量的位置数」
    # `n_eff`，两条规则会在有质量的位置上一致、然后在**零质量位置上各按噪声
    # 任意挑**剩下的名额。实测 J 因此**低**到 0.335（不是趋近 1）——
    # 也就是说这种情况会让人误判成「判据 A 通过，两条规则不同！」，
    # 而那个「不同」全是噪声。**⇒ `k ≥ n_eff` 的区间必须两个方向都不下结论。**
    # `n_eff` = 逐 (层,头) 上按 max 排序后累计到 99% 质量所需的位置数。
    srt = np.sort(mx, axis=-1)[..., ::-1]
    cum = np.cumsum(srt, axis=-1)
    tot = np.maximum(cum[..., -1:], 1e-30)
    n_eff = (cum / tot < 0.99).sum(-1) + 1          # [L,H]

    if not quiet:
        print(f"\n### {label}　U{U.shape}（M={M} 个未来）\n")
        print(f"**有效尾部宽度**：α=0.75 ⇒ k={k75}"
              + ("（**k=1，等于 max，不是 CVaR**）" if k75 == 1 else "")
              + f"；α=0.60 ⇒ k={k2}\n")
        print(f"**n_eff（累计 99% 质量所需位置数）**：中位 {np.median(n_eff):.0f} / "
              f"{N}　—— `k ≥ n_eff` 的头会在零质量位置上按噪声乱挑，"
              f"**那一行的 J 两个方向都不可读**\n")
        print("| ρ | 每头保留 k | **A. J(mean, max)** | A2. J(mean, tail k=%d) | "
              "C. Sp(mean,max) 全体 | C'. 争议池内 | **k≥n_eff 的头占比** |" % k2)
        print("|---|---|---|---|---|---|---|")

    res = {}
    for rho in RHOS:
        k = max(1, int(round(rho * N)))
        ja = jaccard(topk_mask(mean, k), topk_mask(mx, k)).ravel()
        j2 = jaccard(topk_mask(mean, k), topk_mask(t2, k)).ravel()
        sp_all = spearman(mean.reshape(-1, N), mx.reshape(-1, N))
        # 争议池：按 max 取 top-(CONTEND·k)。全体位置里绝大多数对所有未来都 ≈0，
        # 在那里算 argmax / Spearman 是在测噪声。
        kc = min(N, CONTEND * k)
        pool = topk_mask(mx, kc).reshape(-1, N)
        mf, xf = mean.reshape(-1, N), mx.reshape(-1, N)
        sp_c = np.array([spearman(mf[r][pool[r]][None, :], xf[r][pool[r]][None, :])[0]
                         for r in range(mf.shape[0])])
        if not quiet:
            _tr = float((k >= n_eff).mean())
            print(f"| {rho} | {k} | **{ja.mean():.4f}** | {j2.mean():.4f} | "
                  f"{sp_all.mean():+.4f} | {sp_c.mean():+.4f} | "
                  f"{_tr:.3f}{' **⚠不可读**' if _tr > 0.5 else ''} |")
        res[rho] = dict(J=float(ja.mean()), J2=float(j2.mean()),
                        sp=float(sp_all.mean()), sp_c=float(sp_c.mean()), k=k,
                        triv=float((k >= n_eff).mean()))

    # B：多模态。全体 vs 争议池两栏（问题②）
    k = max(1, int(round(0.10 * N)))
    pool = topk_mask(mx, min(N, CONTEND * k))
    arg = U.argmax(0)
    sh_all = np.array([(arg == m).mean() for m in range(M)])
    sh_c = np.array([((arg == m) & pool).sum() / max(pool.sum(), 1) for m in range(M)])
    conc = mx / np.maximum(mean * M, 1e-30)
    if not quiet:
        print(f"\n**B. 各未来当「最需要者」的位置占比**（均匀应为 {1/M:.3f}）")
        print("| 池 | " + " | ".join(f"m{i}" for i in range(M)) + " | max |")
        print("|---|" + "---|" * (M + 1))
        print("| 全体位置 | " + " | ".join(f"{v:.3f}" for v in sh_all)
              + f" | **{sh_all.max():.3f}** |")
        print("| **争议池（top-3k by max）** | " + " | ".join(f"{v:.3f}" for v in sh_c)
              + f" | **{sh_c.max():.3f}** |")
        print(f"\n**B'. 集中度 `max/(M·mean)`**：全体 {conc.mean():.4f} ・ "
              f"争议池 {conc[pool].mean():.4f}"
              f"（{1/M:.3f}=完全均匀，1.0=单一未来独占）")

        # 机制可见：两条规则各自独占的位置，长什么样
        km, kx = topk_mask(mean, k), topk_mask(mx, k)
        only_m, only_x = km & ~kx, kx & ~km
        if only_m.any() and only_x.any():
            print(f"\n**两条规则的分歧长什么样**（ρ=0.10）：")
            print(f"- 只被 mean 选中的位置：集中度 {conc[only_m].mean():.4f} "
                  f"⇒ 多个未来都**弱**需要它")
            print(f"- 只被 max 选中的位置：集中度 {conc[only_x].mean():.4f} "
                  f"⇒ 单个未来**强**需要它")

        print()
        nontriv = {r: v for r, v in res.items()
                   if isinstance(r, float) and v["triv"] <= 0.5}
        # **只用非平凡的 ρ 下判词。** `k ≥ n_eff` 的行里，两条规则在零质量位置上
        # 按噪声乱挑名额，J 无论高低都不反映规则本身（自测⑦）。
        Jmin = min((v["J"] for v in nontriv.values()), default=float("nan"))
        Jmax = max((v["J"] for v in nontriv.values()), default=float("nan"))
        if not nontriv:
            print("⇒ **判据 A 无法执行**：所有 ρ 上多数头都满足 `k ≥ n_eff`，"
                  "两条规则的差异全部来自零质量位置上的噪声挑选，"
                  "**J 两个方向都不可读**。必须换更小的 ρ 或更长的未来窗口后重跑。"
                  "**既不得据此判否，也不得据此宣布通过。**")
        elif Jmin >= J_DEAD:
            print(f"⇒ **判据 A 判否：全部 ρ 上 J(mean,max) ∈ [{Jmin:.4f},{Jmax:.4f}] "
                  f"≥ {J_DEAD}，且非平凡的 ρ 有 {sorted(nontriv)}** ⇒ 两条规则选出"
                  "同一个保留集，ProMeta 的核心命题在本工作点上没有内容。"
                  "**停止，不要写 Student。**")
        elif Jmax >= J_DEAD:
            print(f"⇒ **判据 A 结论随预算变号**（非平凡 ρ={sorted(nontriv)} 上 "
                  f"J ∈ [{Jmin:.4f},{Jmax:.4f}]）⇒ 不是结论，必须按 ρ 分别讨论。")
        else:
            print(f"⇒ **判据 A 通过：非平凡 ρ={sorted(nontriv)} 上 "
                  f"J(mean,max) ≤ {Jmax:.4f} < {J_DEAD}** "
                  "⇒ 两条规则确实选出不同的保留集。**⚠ 这只是必要条件：**"
                  "「不同」≠「更好」，下一步必须做 Oracle-Mean vs Oracle-Tail 的"
                  "**下游任务分数**对照，否则是第⑦类错。")
        if sh_c.max() > 0.9:
            print(f"⇒ **判据 B 判否：争议池内单个未来占 {sh_c.max():.3f}** "
                  "⇒ 未来需求不是多模态的。")
    res["share_max_all"] = float(sh_all.max())
    res["share_max_pool"] = float(sh_c.max())
    res["k75"] = k75
    return res


def selftest():
    rng = np.random.default_rng(0)
    M, L, H, N = 5, 3, 4, 512
    print("=" * 70 + "\n对照（判据本身也要能拒）\n" + "=" * 70)
    same = np.repeat(rng.random((1, L, H, N)), M, axis=0)
    r = analyse(same, "① 五个未来完全相同 ⇒ A 必须判否")
    assert all(r[x]["J"] > 0.999 for x in RHOS), r
    ind = rng.random((M, L, H, N))
    r = analyse(ind, "② 五个未来彼此独立 ⇒ A 必须通过", quiet=True)
    assert all(r[x]["J"] < 0.9 for x in RHOS), r
    print("\n② 独立未来：J = " + ", ".join(f"ρ={x}:{r[x]['J']:.3f}" for x in RHOS)
          + "　PASS（全 <0.9）")
    sp1 = rng.random((M, L, H, N)) * 0.01; sp1[0] += 1.0
    r = analyse(sp1, "③ 只有 m0 主导 ⇒ B 必须判否", quiet=True)
    assert r["share_max_pool"] > 0.9, r
    print(f"③ 单峰未来：争议池内占比 {r['share_max_pool']:.3f}　PASS（>0.9）")
    # ④ tail_mean 的 k 与退化
    _, k = tail_mean(np.zeros((5, 2)), 0.75); assert k == 1, k
    _, k = tail_mean(np.zeros((5, 2)), 0.60); assert k == 2, k
    _, k = tail_mean(np.zeros((20, 2)), 0.75); assert k == 5, k
    print("④ tail_mean 有效 k：M=5,α=.75→1（=max）；M=5,α=.6→2；M=20,α=.75→5　PASS")
    # ⑤ topk_mask 计数与并列
    s = np.zeros((2, 7)); s[0, [1, 3]] = 1.0
    m = topk_mask(s, 2)
    assert m.sum(-1).tolist() == [2, 2] and m[0, 1] and m[0, 3], m
    print("⑤ topk_mask 计数与并列　PASS")
    # ⑥ 争议池确实改变 B（否则限制毫无作用）—— 造一个「噪声位置多」的例子
    # ⑥ 争议池限制必须把「噪声位置稀释多模态」这件事纠正回来。
    # 夹具让争议位置数 ≈ 3k（k=round(.1*512)=51 ⇒ 153），池才近乎纯净。
    U = rng.random((M, L, H, N)) * 1e-6
    U[..., :153] = rng.random((M, L, H, 153)) * 1e-3
    U[0, ..., :153] += 1.0
    r = analyse(U, "⑥ 争议池", quiet=True)
    assert r["share_max_pool"] > 0.9 > r["share_max_all"], r
    print(f"⑥ 争议池限制生效：全体 {r['share_max_all']:.3f} → 池内 "
          f"{r['share_max_pool']:.3f}　PASS（池内 >0.9、全体 <0.9）")
    # ⑦ 平凡一致必须被识别：只有 20 个位置有质量，而 ρ=0.2 ⇒ k=102 ≫ 20
    U = rng.random((M, L, H, N)) * 1e-9
    U[..., :20] = rng.random((M, L, H, 20))
    r = analyse(U, "⑦ 平凡一致", quiet=True)
    # ⚠ 首版这里断言 `J > 0.95`，**错了**：k≫n_eff 时两条规则并不是平凡地一致，
    # 而是在零质量位置上各按噪声乱挑 ⇒ J 反而**低**（实测 0.335）。
    # 这个失效模式比我预想的更糟：它会让人误判成「判据 A 通过」。
    # 正确的断言是「该行被标为不可读」，而不是「J 高」。
    assert r[0.20]["triv"] > 0.9, r[0.20]
    assert r[0.02]["triv"] < 0.5, r[0.02]
    assert r[0.20]["J"] < 0.95, ("k≫n_eff 时 J 应由噪声主导而偏低", r[0.20])
    print(f"⑦ 不可读区间被标出：ρ=0.20 时 k≥n_eff 占比 {r[0.20]['triv']:.3f}、"
          f"J={r[0.20]['J']:.3f}（噪声主导、偏低，**不是**平凡的高）；"
          f"ρ=0.02 时占比 {r[0.02]['triv']:.3f}　PASS")
    print("\nALL PASS")


def main():
    if "--selftest" in sys.argv:
        selftest(); return
    files = [a for a in sys.argv[1:] if a.endswith(".npz")] or \
        sorted(glob.glob("scratch_prometa_oracle_*.npz"))
    if not files:
        print("没有 npz —— 先跑 scratch_prometa_oracle.py。**这不是通过，是没数据。**")
        return
    by_panel = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        m = re.match(r"scratch_prometa_oracle_(.+)_\d+\.npz", os.path.basename(f))
        panel = m.group(1) if m else "unknown"
        by_panel.setdefault(panel, []).append(d["U"])
        print(f"{os.path.basename(f)}: U{d['U'].shape} n_prefix={d['n_prefix']}")
    for panel, Us in sorted(by_panel.items()):
        # ⚠ **同一面板内样本的未来数 M 可能不同**（实测 `scbench_qa_eng` 有样本
        # 是 7 个 question 而非 5）。**不能截断 M** —— 那会改变被测的未来集合、
        # 让「均值 vs 尾部」的对比在不同样本上问的不是同一个问题。
        # 正确做法是按 M 分组，各组独立出表。
        by_m = {}
        for u in Us:
            by_m.setdefault(u.shape[0], []).append(u)
        if len(by_m) > 1:
            print(f"\n**⚠ {panel} 的样本未来数不齐**："
                  + "、".join(f"M={m} 有 {len(v)} 个样本" for m, v in sorted(by_m.items()))
                  + " ⇒ **按 M 分组各自出表，不合并**（截断 M 会改变被测对象）。")
        for m, group in sorted(by_m.items()):
            n = min(u.shape[-1] for u in group)
            # 沿层轴拼样本：每行是一个独立的 (样本,层,头) 单元，逐行统计后再平均
            U = np.concatenate([u[..., :n] for u in group], axis=1)
            analyse(U, f"**{panel}**（M={m}，{len(group)} 个样本）")
    if len(by_panel) < 2:
        print("\n**⚠ 只读到一个面板。** 未来**互不相交**时 `mean = max/M` 是单调变换、"
              "排序必然相同 ⇒ 判据 A 会平凡判否；分歧只可能来自「私有强需求」与"
              "「共享弱需求」并存。`scbench_kv`（5 个不相交 key 查找）**偏悲观**、"
              "`scbench_qa_eng`（同文档 5 个问题，共享背景）**偏乐观** —— "
              "**两个都要跑。**")


if __name__ == "__main__":
    main()
