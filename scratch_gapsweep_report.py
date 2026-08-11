"""B 批（gap 目标三 ckpt × 9 数据集）的配对显著性报告，支持**未跑完的中途快照**。

为什么要支持中途快照：单个 job 最长 5.8h，而三个 ckpt 的进度不同步，等全齐才看会浪费
半天。基线（tag `_full`）与三个变体都从 index 0 顺序跑同一批样本，所以截到**共同样本数**
就仍是严格配对。截断数会打印出来——n 少的时候差值不显著是预期的，别当成"没有效应"。

一律用绝对分 + 按样本配对 bootstrap，不读 results.parse 的相对行：
每个 run 按自己的 full-cache 归一化，而那一档已被证明受空记忆注入污染
（见 CLAUDE.md 2026-08-11 "The memory injects even when it is empty"）。

用法：
    .venv/bin/python -B scratch_gapsweep_report.py --data scbench_prefix_suffix
    .venv/bin/python -B scratch_gapsweep_report.py --data all
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT + "/external/FastKVzip/prefill")
os.chdir(_ROOT + "/external/FastKVzip/prefill")

from results.parse import parse_answer                      # noqa: E402
from results.metric import evaluate_answer                  # noqa: E402

STD = [0.75, 0.5, 0.4, 0.3, 0.2]
M = "qwen2.5-7b-instruct-1m_"
BASE = M + "fastkvzip__full_chunk16k_w4096"
VARIANTS = [
    ("gapf dist (固定0.3)", M + "fastkvzip_gfsd_chunk16k_w4096_varikvdist16_res"),
    ("gapr dist (随机比例)", M + "fastkvzip_grsd_chunk16k_w4096_varikvdist16_res"),
    ("gapr point (随机比例)", M + "fastkvzip_grsp_chunk16k_w4096_varikvpoint16_res"),
]
DATASETS = ["scbench_repoqa", "scbench_prefix_suffix", "scbench_mf", "scbench_vt",
            "scbench_summary", "gsm", "scbench_qa_eng", "squad",
            "scbench_choice_eng"]


def per_sample(data, model_tag, ratios, answ, subt, task="qa", level="pair"):
    """{ratio: [每样本均分]}；逐样本解析照抄 results/parse.py 主循环。"""
    rs = list(ratios) + [1.0]
    out = {r: [] for r in rs}
    i = 0
    while True:
        f = f"./results/{data}/{i}_{model_tag}/output-{level}.json"
        if not os.path.exists(f):
            break
        d = json.load(open(f))
        preds = {r: [] for r in rs}
        answers = []
        for fmt in [k for k in d if k.startswith(task)]:
            for info, text in d[fmt]:
                if info[0] in preds:
                    preds[info[0]].append(text["pruned"])
            if len(preds[1.0]) < len(preds[ratios[-1]]):
                preds[1.0].append(text["full__"])
            answers.append(text["answer"])
        gold = answ[i] if answ else answers
        sub = subt[i] if subt else None
        for r in rs:
            if preds[r]:
                with contextlib.redirect_stdout(io.StringIO()):
                    sc = evaluate_answer(preds[r], gold, data, task, subtask=sub)
                out[r].append(float(np.mean(sc)))
        i += 1
    return out, i


def boot(a, b, n=10000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d.mean(), lo, hi, (lo > 0) or (hi < 0)


def report(data):
    print("=" * 84)
    with contextlib.redirect_stdout(io.StringIO()):
        answ, subt = parse_answer(data)
    base, nb = per_sample(data, BASE, STD, answ, subt)
    if nb == 0:
        print(f"{data}: 基线无结果目录，跳过")
        return
    rows = []
    for name, tag in VARIANTS:
        v, nv = per_sample(data, tag, STD, answ, subt)
        if nv:
            rows.append((name, v, nv))
    if not rows:
        print(f"{data}: 三个变体都还没有结果")
        return
    print(f"{data}   基线 n={nb}")
    for name, v, nv in rows:
        n = min(nb, nv)                     # 截到共同样本数才是严格配对
        flag = "" if nv >= nb else f"  ← 中途快照，截到前 {n} 条（该 job 未跑完）"
        print(f"  {name:22s} n={nv}{flag}")
        for r in STD:
            if not v[r] or not base[r]:
                continue
            a, b = base[r][:n], v[r][:n]
            if len(a) != len(b):
                continue
            m, lo, hi, sep = boot(a, b)
            print(f"      ratio {r:<5} 基线 {np.mean(a)*100:6.2f}  "
                  f"本档 {np.mean(b)*100:6.2f}  Δ {m*100:+7.2f} "
                  f"[{lo*100:+7.2f},{hi*100:+7.2f}] {'★分离' if sep else '未分离'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="all")
    a = ap.parse_args()
    for ds in (DATASETS if a.data == "all" else [a.data]):
        report(ds)
