"""teacher-KL dist 的多数据集报告 —— 绝对分 + 逐样本配对 bootstrap + HRR。

与 `scratch_centroid_report.py` 的区别：那个把 `DATA` 硬编码成 scbench_kv。
这个遍历数据集，两臂都取同批（`_kls_kl` vs `_kls_base`），并把每个数据集的
**满缓存分数**一并报出来 —— headroom 是判读的前提：headroom ≤1 分的数据集上
「补回被驱逐的信息」在算术上就没有目标可打，那里的 Δ≈0 不构成反驳。

用法：
    .venv/bin/python scratch_klsweep_report.py
    .venv/bin/python scratch_klsweep_report.py --partial   # 允许两臂样本数不等，取交集
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
os.chdir(os.path.join(ROOT, "external/FastKVzip/prefill"))
from results.parse import parse_answer, evaluate_answer          # noqa: E402

MODEL = "qwen2.5-7b-instruct-1m"
RATIO = 0.1
# 论文图 11 的面板名 ↔ 数据集 id（CLAUDE.md 的映射表）。**报告一律用面板名**，
# 因为论文里 grep "scbench_kv" 是找不到的 —— 它叫 Retr.KV。
PANEL = {
    "scbench_kv": "Retr.KV", "scbench_prefix_suffix": "Retr.Prefix-Suffix",
    "scbench_repoqa": "Code.RepoQA", "squad": "SQuAD", "gsm": "GSM8K",
    "scbench_qa_eng": "En.QA", "scbench_choice_eng": "En.MultiChoice",
    "scbench_summary": "En.Summary", "scbench_vt": "Retr.MultiHop",
    "scbench_mf": "Math.Find", "scbench_many_shot": "ICL.ManyShot",
}
CATEGORY = {                       # 论文的三个类别
    "scbench_kv": "检索", "scbench_prefix_suffix": "检索",
    "scbench_repoqa": "检索", "squad": "上下文QA", "gsm": "上下文QA",
    "scbench_qa_eng": "上下文QA", "scbench_choice_eng": "上下文QA",
    "scbench_summary": "高冗余", "scbench_vt": "高冗余",
    "scbench_mf": "高冗余", "scbench_many_shot": "高冗余",
}
# (数据集, ratio-0.2 时的 headroom —— 来自 CLAUDE.md 的 headroom 表)
DATASETS = [
    ("scbench_kv", +23.00), ("scbench_prefix_suffix", +10.80), ("gsm", +7.00),
    ("scbench_choice_eng", +6.95), ("scbench_many_shot", +4.82),
    ("scbench_repoqa", +0.91), ("squad", +0.56), ("scbench_vt", -5.02),
]
_MUTE = contextlib.redirect_stdout(io.StringIO())


def per_sample(data, suffix):
    """→ ({idx: 剪枝分}, {idx: 满缓存分})；照抄 parse.py 的解析与打分。"""
    with _MUTE:
        ANSW, SUBT = parse_answer(data)
    task = "qa"
    pr, fu = {}, {}
    i = 0
    miss = 0
    while miss < 5:                      # 结果目录可能不连续，容忍少量缺口
        f = f"./results/{data}/{i}_{MODEL}_fastkvzip_{suffix}/output-pair.json"
        if not os.path.exists(f):
            miss += 1
            i += 1
            continue
        miss = 0
        try:
            d = json.load(open(f))
        except json.JSONDecodeError:      # 正在写入的文件
            i += 1
            continue
        p, q, answers = [], [], []
        for fmt in [k for k in d if k.startswith(task)]:
            for info, text in d[fmt]:
                if abs(float(info[0]) - RATIO) < 1e-9:
                    p.append(text["pruned"])
                    q.append(text["full__"])
            answers.append(text["answer"])
        gold = ANSW[i] if ANSW else answers
        sub = SUBT[i] if SUBT else None
        if p:
            with _MUTE:
                pr[i] = float(np.mean(evaluate_answer(p, gold, data, task, subtask=sub)))
                fu[i] = float(np.mean(evaluate_answer(q, gold, data, task, subtask=sub)))
        i += 1
    return pr, fu


def boot(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    dif = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(dif), (n, len(dif)))
    s = dif[idx].mean(1)
    return dif.mean(), float(np.quantile(s, .025)), float(np.quantile(s, .975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()

    print("=" * 118)
    print(f"teacher-KL dist（ckpt_kl/dist）vs 同批 ratio-{RATIO} 基线　"
          f"★ = 95% CI 不含 0")
    print(f"{'论文面板':<20}{'类别':<9}{'n':>4}{'满缓存':>8}{'基线':>8}"
          f"{'headroom':>9}{'KL dist':>9}{'HRR':>8}{'  Δ (95% CI)':>26}")
    print("-" * 118)
    rows = []
    for ds, hr02 in DATASETS:
        kl, klf = per_sample(ds, "_kls_kl_chunk16k_w4096_varikvdist16_res")
        ba, baf = per_sample(ds, "_kls_base_chunk16k_w4096")
        if ds == "scbench_kv":            # 这个数据集是另一批 tag 跑的
            kl, klf = per_sample(ds, "_klres_dist_chunk16k_w4096_varikvdist16_res")
            ba, baf = per_sample(ds, "_b01_chunk16k_w4096")
        common = sorted(set(kl) & set(ba))
        if not common or (not args.partial and (len(kl) != len(ba))):
            print(f"{PANEL[ds]:<20}{CATEGORY[ds]:<9}{len(common):>4}"
                  f"   —— 未跑完（kl {len(kl)} / base {len(ba)}）"
                  + ("" if common else "，无共同样本"))
            continue
        k = np.array([kl[i] for i in common]) * 100
        b = np.array([ba[i] for i in common]) * 100
        full = np.mean([baf[i] for i in common]) * 100
        head = full - b.mean()
        m, lo, hi = boot(k, b)
        star = "★" if (lo > 0 or hi < 0) else " "
        # headroom ≤0 时 HRR 无意义（除以负数会得到"199%"这种荒谬值）
        hrr_s = f"{m / head * 100:>6.1f}%" if head > 1.0 else "     —"
        print(f"{PANEL[ds]:<20}{CATEGORY[ds]:<9}{len(common):>4}{full:>8.2f}"
              f"{b.mean():>8.2f}{head:>9.2f}{k.mean():>9.2f}{hrr_s:>8}"
              f"{m:>+13.2f} [{lo:+6.2f},{hi:+6.2f}]{star}")
        rows.append((ds, len(common), full, b.mean(), head, k.mean(), m, lo, hi))
    print("=" * 118)
    if rows:
        sep_p = sum(1 for r in rows if r[7] > 0)
        sep_n = sum(1 for r in rows if r[8] < 0)
        print(f"已完成 {len(rows)} 个数据集：显著正 {sep_p}，显著负 {sep_n}，"
              f"未分离 {len(rows)-sep_p-sep_n}；Δ 均值 {np.mean([r[6] for r in rows]):+.2f}")
        print("判读：headroom ≤1 分的数据集上 Δ≈0 不构成反驳（没有目标可打）；"
              "**headroom 为负的 Retr.MultiHop 上显著掉分才是真缺陷**")
        print("（数据集 id ↔ 论文面板：" +
              "  ".join(f"{v}={k}" for k, v in list(PANEL.items())[:4]) + " …）")


if __name__ == "__main__":
    main()
