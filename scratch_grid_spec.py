#!/usr/bin/env python3
"""静态 `u` 表全网格的**唯一真源**：panel 表、ratio 表、tag 构造规则。

为什么单开一个模块：`scratch_fill_grid.py`（排队）与 `scratch_utab_report.py`
（出表）此前**各抄了一份**格子清单，于是排队排出去的新格不会自动进表 ——
KV@0.4 / @0.75 与 GSM8K@0.1 就这样在表外躺了一轮。CLAUDE.md 早写过
「任何选项列表都要从源头派生，不要手抄第二份」（`--arch` 那次 12 个训练
被 argparse 静默拒掉）。这里补上。

tag 规则：`<前缀><panel 短码><ratio 码>`，例如 `_uq01` + `r` + `02` = `_uq01r02`。
"""

# (dataset, 短码, 满量, 显示名, 任务族)
PANELS = [
    ("scbench_kv",            "r",   100, "Retr.KV",        "retrieval"),
    ("scbench_prefix_suffix", "psr", 100, "Retr.PrefSuf",   "retrieval"),
    ("scbench_repoqa",        "rq",   88, "Code.RepoQA",    "retrieval"),
    ("scbench_choice_eng",    "ce",   18, "En.MultiChoice", "qa"),
    ("scbench_qa_eng",        "qa",   20, "En.QA",          "qa"),
    ("gsm",                   "gsm", 100, "GSM8K",          "qa"),
    ("squad",                 "sq",  100, "SQuAD",          "qa"),
    ("scbench_mf",            "mf",  100, "Math.Find",      "redund"),
    ("scbench_summary",       "sm",   70, "En.Summary",     "redund"),
    ("scbench_many_shot",     "ms",   54, "ICL.ManyShot",   "redund"),
    ("scbench_vt",            "vt",   90, "Retr.MultiHop",  "redund"),
]
RATIOS = [(0.1, "01"), (0.2, "02"), (0.3, "03"), (0.4, "04"), (0.5, "05"), (0.75, "075")]

CODE = {d: c for d, c, _, _, _ in PANELS}
NUM = {d: n for d, _, n, _, _ in PANELS}
NAME = {d: nm for d, _, _, nm, _ in PANELS}
FAM = {d: f for d, _, _, _, f in PANELS}
RCODE = dict(RATIOS)
# 族的优先级：教师是 key→value 检索，Retrieval 与它同构、Redundancy 最远
FAM_ORDER = {"retrieval": 0, "qa": 1, "redund": 2}


def tag(ds, ratio, pre="_uq01"):
    """→ 该 (panel, ratio) 在给定前缀下的 tag。**唯一构造入口。**"""
    return f"{pre}{CODE[ds]}{RCODE[ratio]}"


def all_cells():
    """→ [(dataset, ratio)]，按族序、族内按 panel 在 PANELS 里的先后。"""
    out = []
    for fam in ("retrieval", "qa", "redund"):
        for d, _, _, _, f in PANELS:
            if f != fam:
                continue
            out += [(d, r) for r, _ in RATIOS]
    return out
