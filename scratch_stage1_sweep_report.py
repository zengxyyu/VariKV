"""汇总 stage1 容量扫描结果，并对关键档位对做配对比较。

为什么必须配对：各档评的是同一批样本（顺序确定），tier5 相对 tier2/4 的差距
在 K=16 时只有 0.02~0.07 nats，而样本间 nll 的方差远大于此。非配对的均值差
分辨不出它是真信号还是噪声；配对差的均值 ± 自助法置信区间才能。
"""
import argparse
import json
import math
import random
from pathlib import Path

TIERS = [1, 2, 3, 4, 5]
# tier1 是「滑动窗口+丢弃」，**不是 KVzip**（KVzip 用重建注意力打分，强得多）。
# 早先标成 discard(KVzip) 会让「Δ vs tier1」被误读成赢过 KVzip。见 CLAUDE.md。
NAME = {1: "recency+discard(滑动窗口)", 2: "recency+point", 3: "recency+moment(MomentKV)",
        4: "fe+point(IndexMem)", 5: "fe+dist(VariKV)"}
LEVELS = [0, 200, 800, 2000]


def finite(xs):
    return [x for x in xs if x == x and not math.isinf(x)]


def boot_ci(diffs, n_boot=10000, seed=0):
    """配对差均值的自助法 95% 置信区间。不含 0 才算分离。

    必须先滤 NaN：一个 NaN 会让 mean 变 NaN、让重采样出的分位数乱序，
    从而打印出看似「分离」的假结论（2026-08-07 就这么误报过一轮）。
    """
    diffs = finite(diffs)
    if len(diffs) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (sum(diffs) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def sep_flag(lo, hi):
    if lo != lo or hi != hi:
        return "n/a"
    return "YES" if (lo > 0 or hi < 0) else "no"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 32, 64])
    args = ap.parse_args()
    d = Path(args.logdir)

    data = {}
    for k in args.ks:
        for t in TIERS:
            f = d / f"res_k{k}_tier{t}.json"
            if f.exists():
                try:
                    data[(k, t)] = json.loads(f.read_text())["tiers"][str(t)]
                except Exception as e:
                    print(f"[warn] 解析失败 {f}: {e}")

    if not data:
        print("没有任何结果文件")
        return

    # ---- 表 1：overall ----
    print("\n表1  overall（主指标 nll，越低越好；em 仅供参考）")
    print("-" * 74)
    print(f"{'K':>4} {'tier':>5}  {'方法':<26} {'nll':>8} {'Δ vs t1':>9} {'em':>7} {'n':>5}")
    for k in args.ks:
        base = data.get((k, 1), {}).get("overall", {}).get("nll")
        for t in TIERS:
            r = data.get((k, t))
            if not r:
                continue
            o = r["overall"]
            dv = f"{o['nll'] - base:+8.4f}" if base is not None else "       -"
            print(f"{k:>4} {t:>5}  {NAME[t]:<26} {o['nll']:>8.4f} {dv:>9} "
                  f"{o['em']:>7.3f} {o['n']:>5}")
        print()

    # ---- 表 2：按干扰档 ----
    print("\n表2  按干扰强度分档的 nll")
    print("-" * 74)
    hdr = f"{'K':>4} {'tier':>5} " + "".join(f"{('nd=' + str(l)):>11}" for l in LEVELS)
    print(hdr)
    for k in args.ks:
        for t in TIERS:
            r = data.get((k, t))
            if not r:
                continue
            cells = []
            for l in LEVELS:
                v = r.get(f"all/{l}")
                cells.append(f"{v['nll']:>11.4f}" if v else f"{'-':>11}")
            print(f"{k:>4} {t:>5} " + "".join(cells))
        print()

    # ---- 表 3：配对比较 ----
    print("\n表3  配对比较（负 = 前者更好）。95% 自助置信区间不含 0 才算分离")
    print("-" * 74)
    pairs = [(5, 2), (5, 4), (5, 3), (5, 1)]
    print(f"{'K':>4}  {'对比':<16} {'配对Δnll':>10} {'95% CI':>22} {'分离?':>7}")
    for k in args.ks:
        for a, b in pairs:
            ra, rb = data.get((k, a)), data.get((k, b))
            if not ra or not rb:
                continue
            pa = {(x["i"]): x["nll"] for x in ra.get("_per_sample", [])}
            pb = {(x["i"]): x["nll"] for x in rb.get("_per_sample", [])}
            common = sorted(set(pa) & set(pb))
            if not common:
                continue
            diffs = [pa[i] - pb[i] for i in common]
            n_bad = len(diffs) - len(finite(diffs))
            m, lo, hi = boot_ci(diffs)
            sep = sep_flag(lo, hi)
            note = f"  (丢弃{n_bad}个非有限)" if n_bad else ""
            print(f"{k:>4}  t{a} vs t{b:<12} {m:>10.4f}   [{lo:>8.4f},{hi:>8.4f}] {sep:>7}{note}")
        print()

    # ---- 表 4：容量效应 ----
    print("\n表4  容量效应：同一档位随 K 的变化（配对，相对 K=16）")
    print("-" * 74)
    print(f"{'tier':>5}  {'K':>4} {'配对Δnll vs K=16':>18} {'95% CI':>22} {'分离?':>7}")
    for t in TIERS:
        r16 = data.get((16, t))
        if not r16:
            continue
        p16 = {x["i"]: x["nll"] for x in r16.get("_per_sample", [])}
        for k in args.ks:
            if k == 16:
                continue
            rk = data.get((k, t))
            if not rk:
                continue
            pk = {x["i"]: x["nll"] for x in rk.get("_per_sample", [])}
            common = sorted(set(p16) & set(pk))
            if not common:
                continue
            diffs = [pk[i] - p16[i] for i in common]
            n_bad = len(diffs) - len(finite(diffs))
            m, lo, hi = boot_ci(diffs)
            sep = sep_flag(lo, hi)
            note = f"  (丢弃{n_bad}个非有限)" if n_bad else ""
            print(f"{t:>5}  {k:>4} {m:>18.4f}   [{lo:>8.4f},{hi:>8.4f}] {sep:>7}{note}")
        print()


if __name__ == "__main__":
    main()
