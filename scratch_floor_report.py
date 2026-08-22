#!/usr/bin/env python3
"""地板 `b_min=8` 全 66 格网格的**附加**分析（表本身由
`scratch_utab_report.py --pre _fl8` 生成，本脚本不重复造表）。

本脚本回答四件 `utab_report` 不管的事：

  ① **无操作格的逐位对照**。`[floor]` 日志给出每 chunk 的 `lift`；若某格
     全程 `lift == 0`，地板在数学上什么都没做 ⇒ 该格**必须**与基线逐样本
     完全相同。这是整张网格自带的阳性对照：**它若不成立，说明流水线有
     不确定性，整张表作废**。已实测 Retr.KV@0.75 是 lift=0、@0.5 是 lift=3。
  ② **地板动作量的客观刻画**：`n_starved`（饿死头数）、`lift`（抬起的总配额）、
     `lift/Btot`（占预算比例）。这些是**从运行时日志读出来的**，不是从表推的。
  ③ **Δ 对 starvation 率 / headroom / 相对搬动量的回归** —— 找一个
     **无标签**的部署谓词（推理时拿得到的量）。
  ④ **地板 vs 静态 `u` 表逐样本配对**。

**为什么不并进 `utab_report`**：那个脚本服务三张表（kv/psyn/r05），加进去会让
它对非地板的表打印一堆空列。**判据写成代码、判词由数字生成** —— 每个判据在
`--selftest` 下都有阴性对照。
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scratch_grid_spec import PANELS, RATIOS, tag as mktag   # noqa: E402
from scratch_read_scores import read_scores                  # noqa: E402

PRE = "_fl8"
for _i, _a in enumerate(sys.argv):
    if _a == "--pre" and _i + 1 < len(sys.argv):
        PRE = sys.argv[_i + 1]

# `[floor] chunk lo=28 bmin=8 n_starved=64/112 n_lifted=78 lift=589 Btot=103282`
FLOOR_RE = re.compile(
    r"\[floor\] chunk lo=(\d+) bmin=(\d+) n_starved=(\d+)/(\d+) "
    r"n_lifted=(\d+) lift=([0-9.]+) Btot=(\d+)")


def floor_stats(tag):
    """→ dict 或 None。从运行时日志聚合地板的实际动作。

    **必须读日志而不是重算**：`b0` 依赖真实的 chunk 边界与分数分布，
    离线重算等于把「我以为它做了什么」当成「它做了什么」（第六类错）。
    """
    lg = os.path.join("scratch_ctrl_logs", tag.lstrip("_") + ".log")
    if not os.path.exists(lg):
        return None
    rows = FLOOR_RE.findall(open(lg, errors="ignore").read())
    if not rows:
        return None
    bmin = {int(r[1]) for r in rows}
    st = np.array([int(r[2]) for r in rows], dtype=float)
    ng = np.array([int(r[3]) for r in rows], dtype=float)
    lf = np.array([float(r[5]) for r in rows])
    bt = np.array([int(r[6]) for r in rows], dtype=float)
    # **需求/预算 = b_min·L·H / Btot** —— 跨 panel 比较前必须看这一列。
    # 2026-08-22 的教训：`b_min=8` 在 Retr.KV 上占预算 0.64%、在 GSM8K@0.1 上
    # 占 **109%**（不可行，触发 `min(b_min, Btot//(L·H))` 降级 ⇒ 近似均匀分配）。
    # 同一个绝对数在两类 panel 上是相差 125 倍的干预（第②类错）。
    bm0 = max(bmin) if bmin else 0
    ngp = int(ng[0])
    med_bt = float(np.median(bt))
    demand = (bm0 * ngp / med_bt) if med_bt > 0 else float("inf")
    sat = float((bt // max(ngp, 1) < bm0).mean())     # 触发降级的 chunk 占比
    return dict(n_chunks=len(rows), bmin=sorted(bmin),
                starve_frac=float((st / ng).mean()), n_groups=ngp,
                lift_total=float(lf.sum()), lift_max=float(lf.max()),
                lift_frac=float((lf / np.maximum(bt, 1)).mean()),
                btot_med=med_bt, demand=demand, sat_frac=sat)


def finished(tag):
    lg = os.path.join("scratch_ctrl_logs", tag.lstrip("_") + ".log")
    return os.path.exists(lg) and "Finished." in open(lg, errors="ignore").read()


def cell(ds, tag, r, n_full):
    """→ (base, arm, d_vec) 或 None。完整性硬闸与 `utab_report` 同口径。"""
    if not finished(tag):
        return None
    try:
        b = read_scores(ds, "__g8base", r)
        v = read_scores(ds, tag, r)
    except Exception:
        return None
    k = sorted(set(b) & set(v))
    if not k or len(k) != len(b) or len(b) != n_full:
        return None
    return (np.array([b[i] for i in k]), np.array([v[i] for i in k]),
            np.array([v[i] - b[i] for i in k]))


def boot(d, rng):
    bs = d[rng.integers(0, len(d), size=(4000, len(d)))].mean(1) * 100
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def check_noop(rows):
    """判据 ①：`lift_total == 0` 的格必须逐样本与基线完全相同。

    **阴性对照在 `--selftest` 里**：喂一个 lift=0 但分数不同的假格，
    必须报 FAIL —— 判据本身也要自测（栽过的第一类错）。
    """
    out = []
    for r in rows:
        if r["fs"] is None or r["fs"]["lift_total"] != 0.0:
            continue
        same = bool(np.all(r["d"] == 0))
        out.append((r["nm"], r["rho"], same, float(np.abs(r["d"]).max()) * 100))
    return out


def main():
    rng = np.random.default_rng(0)
    full = {}
    for ds, _c, _n, nm, _f in PANELS:
        try:
            full[ds] = np.mean(list(read_scores(ds, "__g8base", 1.0).values())) * 100
        except Exception:
            full[ds] = float("nan")

    rows = []
    for ds, _c, n_full, nm, fam in PANELS:
        for r, _rc in RATIOS:
            tg = mktag(ds, r, PRE)
            c = cell(ds, tg, r, n_full)
            if c is None:
                continue
            b, v, d = c
            lo, hi = boot(d, rng)
            rows.append(dict(ds=ds, nm=nm, fam=fam, rho=r, tag=tg,
                             base=b.mean() * 100, arm=v.mean() * 100,
                             delta=d.mean() * 100, lo=lo, hi=hi, n=len(d), d=d,
                             head=full[ds] - b.mean() * 100,
                             fs=floor_stats(tg)))

    print(f"# 地板 `{PRE}` 附加分析 —— {len(rows)} / "
          f"{len(PANELS) * len(RATIOS)} 格满量\n")
    if not rows:
        print("**没有满量格，全部分析跳过（不是通过，是没数据）。**")
        return

    print("## ① 无操作格的逐位对照（阳性对照）\n")
    noop = check_noop(rows)
    if not noop:
        print("**当前没有 `lift_total == 0` 的满量格 —— 判据无法执行。**\n")
    else:
        print("| panel | ρ | 逐样本全等基线 | max\\|Δ\\| |")
        print("|---|---|---|---|")
        for nm, rho, same, mx in noop:
            print(f"| {nm} | {rho} | {'**是 ✓**' if same else '**否 ✗**'} | {mx:.4f} |")
        bad = [x for x in noop if not x[2]]
        print(f"\n**{len(noop)} 个无操作格，{len(bad)} 个不等 ⇒ "
              + ("**判据通过：流水线确定，表可信。**" if not bad else
                 "**判据失败：地板什么都没做却改了分数 ⇒ 流水线有不确定性，整张表作废。**"))
        print()

    print("## ② 地板的实际动作（从 `[floor]` 运行时日志读出，非重算）\n")
    print("**⚠ 先看「需求/预算」列。** 它 = `b_min·L·H / Btot`（`Btot` 取全 chunk 中位）。"
          "超过 100% 表示地板**在数学上不可行**，`quota_project` 会降级到 "
          "`min(b_min, Btot//(L·H))`，干预实际退化成**近似均匀分配** —— "
          "那样的格**不能与需求 <1% 的格并列比较**，它们跑的不是同一个方法。\n")
    print("| panel | ρ | `Btot` 中位 | **需求/预算** | 降级 chunk 占比 | 饿死头占比 | "
          "抬起总配额 | b_min | 制度 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x["nm"], x["rho"])):
        f = r["fs"]
        if f is None:
            print(f"| {r['nm']} | {r['rho']} | — 无日志 | — | — | — | — | — | — |")
            continue
        reg = ("**⚠不可行/退化**" if f["demand"] > 1.0 else
               "**⚠高剂量**" if f["demand"] > 0.05 else "微调")
        print(f"| {r['nm']} | {r['rho']} | {f['btot_med']:.0f} | "
              f"**{100*f['demand']:.2f}%** | {f['sat_frac']:.2f} | "
              f"{f['starve_frac']:.3f} ({f['n_groups']} 组) | {f['lift_total']:.0f} | "
              f"{f['bmin']} | {reg} |")
    hi = [r for r in rows if r["fs"] and r["fs"]["demand"] > 0.05]
    print(f"\n**需求 >5% 的格：{len(hi)}/{len(rows)}** —— "
          + ("无。全部格同属「微调」制度，可横向比较。"
             if not hi else
             "以下格与其余格**不是同一强度的干预**，横向比较无效：" +
             "、".join(f"{r['nm']}@{r['rho']}({100*r['fs']['demand']:.0f}%)" for r in hi))
          + "\n")

    print("## ③ Δ 对三个**无标签**候选谓词的回归\n")
    print("目标：找一个推理时拿得到的量来决定「何时不干预」。"
          "**headroom 需要满缓存参照，推理时拿不到 —— 它是参照上界不是候选。**\n")
    preds = [("饿死头占比", lambda r: r["fs"]["starve_frac"] if r["fs"] else np.nan),
             ("抬起占预算比", lambda r: r["fs"]["lift_frac"] if r["fs"] else np.nan),
             ("headroom（非无标签，仅参照）", lambda r: r["head"])]
    y = np.array([r["delta"] for r in rows])
    print("| 谓词 | n | Pearson | Spearman | 斜率 | 截距 |")
    print("|---|---|---|---|---|---|")
    for name, fn in preds:
        x = np.array([fn(r) for r in rows])
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            print(f"| {name} | {int(m.sum())} | — 样本不足 | — | — | — |")
            continue
        xx, yy = x[m], y[m]
        pe = float(np.corrcoef(xx, yy)[0, 1])
        rx = np.argsort(np.argsort(xx)).astype(float)
        ry = np.argsort(np.argsort(yy)).astype(float)
        sp = float(np.corrcoef(rx, ry)[0, 1])
        a, b0 = np.polyfit(xx, yy, 1)
        print(f"| {name} | {int(m.sum())} | {pe:+.3f} | {sp:+.3f} | "
              f"{a:+.2f} | {b0:+.2f} |")
    print()

    print("## ④ 失败格是否都落在 headroom < 0 的 panel\n")
    neg = [r for r in rows if r["hi"] < 0]
    print(f"显著为负 **{len(neg)}** 格。\n")
    if neg:
        print("| panel | ρ | Δ | headroom | headroom<0 |")
        print("|---|---|---|---|---|")
        for r in sorted(neg, key=lambda x: x["delta"]):
            print(f"| {r['nm']} | {r['rho']} | **{r['delta']:+.2f}** | "
                  f"{r['head']:+.2f} | {'**是**' if r['head'] < 0 else '否'} |")
        allneg = all(r["head"] < 0 for r in neg)
        print(f"\n**{'全部' if allneg else '并非全部'}落在 headroom < 0 的 panel** ⇒ "
              + ("缺口精确等于「何时不干预」。" if allneg
                 else "「负 headroom 解释失败格」这个说法**不成立**。"))
    print()

    print("## ⑤ 地板 vs 静态 `u` 表，逐样本配对\n")
    print("| panel | ρ | 地板 Δ | `u` 表 Δ | 配对差 | 95% CI |")
    print("|---|---|---|---|---|---|")
    npair = 0
    for r in sorted(rows, key=lambda x: (x["nm"], x["rho"])):
        ut = mktag(r["ds"], r["rho"], "_uq01")
        c = cell(r["ds"], ut, r["rho"], r["n"])
        if c is None:
            continue
        _, _, du = c
        if len(du) != len(r["d"]):
            continue
        dd = r["d"] - du
        lo, hi = boot(dd, rng)
        npair += 1
        print(f"| {r['nm']} | {r['rho']} | {r['delta']:+.2f} | "
              f"{du.mean()*100:+.2f} | **{dd.mean()*100:+.2f}**"
              f"{'★' if (lo > 0 or hi < 0) else ' ns'} | [{lo:+.2f}, {hi:+.2f}] |")
    if npair == 0:
        print("| — | — | — | — | 无同格 `u` 表读数 | — |")
    print()


def selftest():
    """判据自测 + 阴性对照。判据本身也会错（第一类错）。"""
    ok = True

    # ① FLOOR_RE 必须解析真实日志行
    line = ("[floor] chunk lo=28 bmin=8 n_starved=64/112 n_lifted=78 "
            "lift=589 Btot=103282")
    m = FLOOR_RE.findall(line)
    assert m and m[0] == ("28", "8", "64", "112", "78", "589", "103282"), m
    print("① 正则解析真实日志行           PASS")

    # ② 阴性对照：格式变了必须解析不出来，而不是悄悄给个默认值
    assert not FLOOR_RE.findall("[floorcov] chunk lo=28 frac=1.0"), "误吞 floorcov"
    assert not FLOOR_RE.findall("[floor] chunk lo=28 bmin=8"), "残行不该匹配"
    print("② 阴性对照：floorcov / 残行不匹配  PASS")

    # ③ check_noop 的阳性 + 阴性
    good = [dict(nm="X", rho=0.75, fs=dict(lift_total=0.0), d=np.zeros(10))]
    bad = [dict(nm="Y", rho=0.75, fs=dict(lift_total=0.0),
                d=np.array([0.0] * 9 + [0.02]))]
    skip = [dict(nm="Z", rho=0.1, fs=dict(lift_total=589.0), d=np.ones(10))]
    r1, r2, r3 = check_noop(good), check_noop(bad), check_noop(skip)
    assert len(r1) == 1 and r1[0][2] is True, r1
    assert len(r2) == 1 and r2[0][2] is False, r2
    assert r3 == [], "有动作的格不该进无操作判据"
    print("③ check_noop 阳性/阴性/跳过        PASS")

    # ④ 格子清单必须来自唯一真源，且恰好 66
    n = len(PANELS) * len(RATIOS)
    assert n == 66, n
    assert len({mktag(d, r, PRE) for d, *_ in PANELS for r, _ in RATIOS}) == 66
    print(f"④ grid_spec 派生 {n} 格、tag 无碰撞   PASS")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        print("\nALL PASS")
    else:
        main()
