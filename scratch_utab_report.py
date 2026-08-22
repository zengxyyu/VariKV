"""静态 `u` 表全部评测格的汇总表 —— **由原始结果重算，不手抄**（2026-08-21）。

为什么要一个生成器：本文件里的数字被反复引用，手抄进文档已经出过错（撤回 57
就是把两个架构的数混比）。这里从 `results/` 逐样本重算、逐格配对 bootstrap，
输出 markdown 直接贴进 `RESULTS_ABLATION.md`。**改了就重跑，别手改表。**

口径：Δ = 与**同 ratio** 的 `__g8base` 基线的逐样本配对差；★ = 95% bootstrap CI
排除 0；headroom = 满缓存 − 同 ratio 基线（满缓存取 ρ=1.0 的基线）。
"""
import os
import sys
import numpy as np

sys.path.insert(0, ".")
from scratch_read_scores import read_scores          # noqa: E402

# **panel / ratio / tag 全部从 `scratch_grid_spec` 派生**（2026-08-21 改）。
# 此前这里手抄了一份 `{panel: {ratio: tag}}`，而排队脚本另抄一份 ==> 新排的格
# 跑完也不会进表：KV@0.4 / @0.75 与 GSM8K@0.1 就这样在表外躺了一轮，
# 磁盘上有 32 格而表里只列 19 格。CLAUDE.md 早写过「选项列表要从源头派生，
# 不要手抄第二份」，这里补上。
from scratch_grid_spec import PANELS, RATIOS, NAME, tag as _mktag   # noqa: E402

PANEL = [(d, NAME[d]) for d, _, _, _, _ in PANELS]
RAT = [r for r, _ in RATIOS]
# `--pre` 让同一个生成器同时服务多张表（kv 表 `_uq01`、psyn 表 `_up01`、
# ρ=0.5 表 `_ur5`）。**只有一份格子清单与一份口径**，不再手抄第二份。
PRE = "_uq01"
for _i, _a in enumerate(sys.argv):
    if _a == "--pre" and _i + 1 < len(sys.argv):
        PRE = sys.argv[_i + 1]
UTAB = {d: {r: _mktag(d, r, PRE) for r in RAT} for d, _, _, _, _ in PANELS}
CTRL = [("取负 −u @Retr.KV", "scbench_kv", 0.1, "_uqneg01r01"),
        ("取负 −u @Retr.KV", "scbench_kv", 0.5, "_uqneg01r05"),
        ("置换 @Retr.KV", "scbench_kv", 0.1, "_uqperm01r01"),
        ("取负 −u @Retr.MultiHop", "scbench_vt", 0.2, "_uqneg01vt02"),
        ("剂量 4× @Retr.KV", "scbench_kv", 0.5, "_uq004r05")]


def cell(ds, tag, r):
    """→ (基线均值, 臂均值, Δ, lo, hi, n) 全部 ×100；不完整或读不到返回 None。

    **完整性硬闸（本项目铁律，别去掉）**：只有日志里出现 `Finished.` **且**样本数
    等于同 panel 基线的样本数时才返回结果。理由是记录在案的坑：Math.Find 在
    38/100 时读到 −3.95★，满量后是 −2.33 不显著 —— **partial samples 上的 ★
    一律不可信**。首版没有这道闸，把三个在跑的 PrefSuf 格（n=94/69/58）和
    GSM8K（n=88）当成了结果。
    """
    lg = os.path.join("scratch_ctrl_logs", tag.lstrip("_") + ".log")
    if not os.path.exists(lg) or "Finished." not in open(lg, errors="ignore").read():
        return None
    try:
        b = read_scores(ds, "__g8base", r)
        v = read_scores(ds, tag, r)
    except Exception:
        return None
    k = sorted(set(b) & set(v))
    if not k or len(k) != len(b):          # 样本数必须与基线相等
        return None
    d = np.array([v[i] - b[i] for i in k])
    rs = np.random.default_rng(0)
    bs = d[rs.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # bootstrap 双侧 p：BH-FDR 需要它。66 格各用名义 95% CI 期望约 3.3 个假阳，
    # 所以任何「全网格无显著负值」这类全称判断都必须报校正后的版本。
    pv = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
    pv = min(1.0, max(pv, 1.0 / len(bs)))      # 0 不可用于 BH，压到分辨率下限
    return (np.mean([b[i] for i in k]) * 100, np.mean([v[i] for i in k]) * 100,
            d.mean() * 100, lo * 100, hi * 100, len(k), pv)


def fmt(c):
    if c is None:
        return "—"
    _, _, m, lo, hi, _, *_ = c
    return f"{m:+.2f}{'★' if (lo > 0 or hi < 0) else ' ns'}"


def _bh(pvs, q=0.05):
    """Benjamini-Hochberg，按方向分别报。

    判据 4 要求的就是这一段：66 格各用名义 95% CI 时期望约 3.3 个假阳，
    「全网格无显著负值」这种全称判断只有在校正后才站得住。
    正负分开报是因为这两个方向在本项目里的含义完全不同 ——
    负值关的是安全性（能不能部署），正值关的是有效性。
    """
    if not pvs:
        print("**BH-FDR：无满量格，跳过。**\n")
        return
    m = len(pvs)
    order = sorted(range(m), key=lambda i: pvs[i][3])
    thr, kmax = None, -1
    for rank, i in enumerate(order, 1):
        if pvs[i][3] <= q * rank / m:
            kmax, thr = rank, q * rank / m
    keep = set(order[:kmax]) if kmax > 0 else set()
    pos = [pvs[i] for i in keep if pvs[i][2] > 0]
    neg = [pvs[i] for i in keep if pvs[i][2] < 0]
    print(f"**BH-FDR（q=0.05，m={m} 格）：校正后显著为正 {len(pos)} 格 ・ "
          f"显著为负 {len(neg)} 格 ・ 其余 {m - len(keep)} 格不可分。**")
    if neg:
        print("\n校正后仍显著为负的格（安全性判据的反例）：\n")
        print("| panel | ρ | Δ | p |")
        print("|---|---|---|---|")
        for nm, r, d, pv in sorted(neg, key=lambda x: x[2]):
            print(f"| {nm} | {r} | **{d:+.2f}** | {pv:.4f} |")
    print()


def main():
    print(f"| panel | ρ | headroom | 基线 | 表 `{PRE}` | **Δ** | 95% CI | n |")
    print("|---|---|---|---|---|---|---|---|")
    nstar_pos = nstar_neg = nns = 0
    _pvs = []
    for ds, nm in PANEL:
        try:
            full = np.mean(list(read_scores(ds, "__g8base", 1.0).values())) * 100
        except Exception:
            full = float("nan")
        for r in RAT:
            tag = UTAB.get(ds, {}).get(r)
            if not tag:
                continue
            c = cell(ds, tag, r)
            if c is None:
                continue
            base, arm, m, lo, hi, n, *_rest = c
            _pvs.append((nm, r, m, _rest[0] if _rest else 1.0))
            if lo > 0:
                nstar_pos += 1
            elif hi < 0:
                nstar_neg += 1
            else:
                nns += 1
            print(f"| {nm} | {r} | {full - base:+.2f} | {base:.2f} | {arm:.2f} | "
                  f"**{m:+.2f}**{'★' if (lo > 0 or hi < 0) else ' ns'} | "
                  f"[{lo:+.2f}, {hi:+.2f}] | {n} |")
    print(f"\n**统计（名义 95% CI）：显著为正 {nstar_pos} 格 ・ 显著为负 "
          f"{nstar_neg} 格 ・ 不可分 {nns} 格。**\n")
    _bh(_pvs)
    if PRE != "_uq01":
        return          # 对照组是给 kv 表登记的，换表时不重复打印
    print("| 对照（只变一个变量） | panel | ρ | Δ | 95% CI | n |")
    print("|---|---|---|---|---|---|")
    for name, ds, r, tag in CTRL:
        c = cell(ds, tag, r)
        if c is None:
            print(f"| {name} | {ds} | {r} | **在跑/已排** | — | — |")
            continue
        _, _, m, lo, hi, n, *_ = c
        print(f"| {name} | {ds} | {r} | **{m:+.2f}**"
              f"{'★' if (lo > 0 or hi < 0) else ' ns'} | [{lo:+.2f}, {hi:+.2f}] | {n} |")


if __name__ == "__main__":
    main()
