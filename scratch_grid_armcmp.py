#!/usr/bin/env python3
"""从 `RESULTS_GRID.md` **机器解析**任意几条臂的跨 ratio 对照。

**为什么要有这个脚本**：2026-08-22 我在只看 `Retr.KV@0.1` 一格的情况下断言
「`chead10` 是 `scalar` 的 5 倍」，而全网格上 `chd10` 五个 panel 全负、
`scalar` 五个全正。**数据当时就在 `RESULTS_GRID.md` 里**（用户打开的正是那个
文件），我没查 —— 违反了自己写的两条规则：「写新分析前先 `command grep` 文档」
与第⑧类错「只看一个工作点就下判断」。

所以这个脚本存在的意义是**让「看全网格」变成一条命令**，而不是一件要靠自觉
去做的事。`RESULTS_GRID.md` 由 `scratch_all_report.py` 生成，是唯一真源，
本脚本只读不写。

    .venv/bin/python scratch_grid_armcmp.py scalar chd10 chead sgs
"""
import re
import sys

SRC = "RESULTS_GRID.md"
ORD = ["ρ=0.75", "ρ=0.5", "ρ=0.4", "ρ=0.3", "ρ=0.2", "ρ=0.1", "ρ=0.05"]


def parse(path=SRC):
    """→ {panel: {arm: {ratio: (Δ, is_star)}}}。`—`（没跑）解析成 None，
    **不是 0** —— 把没跑当成零是这个项目栽过的坑。"""
    head, panel, out = None, None, {}
    for l in open(path).read().splitlines():
        if l.startswith("| panel") and "ρ=" in l:
            head = [c.strip() for c in l.strip("|").split("|")][3:]
            continue
        if not l.startswith("|") or head is None:
            continue
        c = [x.strip() for x in l.strip("|").split("|")]
        if len(c) < 4:
            continue
        if c[0] and c[0] != "panel" and not set(c[0]) <= set("- "):
            panel = c[0]
        arm = c[2]
        if arm in ("arm", "") or panel is None:
            continue
        v = {}
        for r, x in zip(head, c[3:]):
            m = re.match(r"^([+-][0-9.]+)", x)
            v[r] = (float(m.group(1)), "★" in x) if m else None
        out.setdefault(panel, {})[arm] = v
    return out


def main():
    arms = sys.argv[1:] or ["scalar", "chd10"]
    g = parse()
    print("| panel | 臂 | " + " | ".join(ORD) + " | **合计** | 格数 |")
    print("|---|---|" + "---|" * (len(ORD) + 2))
    tot = {a: [0.0, 0] for a in arms}
    for p, d in g.items():
        if not any(a in d for a in arms):
            continue
        for a in arms:
            if a not in d:
                continue
            cells, s, n = [], 0.0, 0
            for r in ORD:
                x = d[a].get(r)
                if x is None:
                    cells.append("—")
                    continue
                cells.append(f"{x[0]:+.2f}{'★' if x[1] else ''}")
                s += x[0]
                n += 1
            if n == 0:
                continue
            tot[a][0] += s
            tot[a][1] += n
            print(f"| {p} | `{a}` | " + " | ".join(cells) + f" | **{s:+.2f}** | {n} |")
    print()
    print("| 臂 | 全部已测格合计 | 格数 | 均值/格 |")
    print("|---|---|---|---|")
    for a in arms:
        s, n = tot[a]
        if n:
            print(f"| `{a}` | **{s:+.2f}** | {n} | {s/n:+.2f} |")
    print("\n**⚠ 「合计」只是把已测格相加，不是统计量** —— 各格 n 不同、"
          "各 panel 量纲不同、且缺格按 `—` 跳过而非补零。它只用来看**符号与量级**，"
          "不要当成效应量。要做推断请回到逐格的配对 bootstrap。")


if __name__ == "__main__":
    main()
