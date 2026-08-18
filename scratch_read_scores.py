#!/usr/bin/env python3
"""逐样本读分数的**通用**读取器 —— 不假设 gate 前缀，也不假设输出文件名。

为什么需要它：本项目此前的读数片段都硬编码 `*_fastkvzip{tag}` 与 `output-pair.json`。
跨方法移植要比的是

    results/<ds>/<i>_<model>_expect__expbase_.../output-adakv-layer.json      （捐赠方）
    results/<ds>/<i>_<model>_fastkvzip__xpFKVqExp_.../output-pair.json        （接收方）

**gate 前缀与输出文件名两者都不同**，硬编码版会**静默返回空**再打印 0.00 —— 本项目
已经被这一类静默零坑过（`VARIKV_RATIOS` 没导出给 parse 时同样打印 0.00）。所以这里
按 `<tag>_` 匹配目录、按 `output-*.json` 通配输出，并且**匹配到 0 个目录时直接抛错**，
不返回空字典。

用法（import 或命令行）：
    python scratch_read_scores.py scbench_kv _expbase 0.2
"""
import contextlib
import glob
import io
import json
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.dirname(__file__))
_PRE = os.path.join(_ROOT, "external/FastKVzip/prefill")
if _PRE not in sys.path:
    sys.path.insert(0, _PRE)
_MUTE = contextlib.redirect_stdout(io.StringIO())


def read_scores(ds, tag, ratio, model="qwen2.5-7b-instruct-1m", strict=True):
    """→ {sample_idx: score}。`tag` 形如 `_expbase`（前导下划线可有可无）。"""
    from results.parse import parse_answer, evaluate_answer          # noqa: E402
    cwd = os.getcwd()
    os.chdir(_PRE)
    try:
        t = tag if tag.startswith("_") else "_" + tag
        # 目录名形如 <i>_<model>_<gate>_<tag>_chunk...；gate 未知，故用两个通配
        pat = f"results/{ds}/*_{model}_*{t}_*/output-*.json"
        files = sorted(glob.glob(pat))
        if strict and not files:
            raise FileNotFoundError(f"没有匹配的结果目录：{pat}（tag 或 ds 写错？）")
        with _MUTE:
            ans, sub = parse_answer(ds)
        out = {}
        for f in files:
            i = int(os.path.basename(os.path.dirname(f)).split("_")[0])
            try:
                d = json.load(open(f))
            except Exception:
                continue
            pred, gold = [], []
            for k in [x for x in d if x.startswith("qa")]:
                for info, rec in d[k]:
                    if abs(float(info[0]) - ratio) < 1e-9:
                        pred.append(rec["pruned"]); gold.append(rec["answer"])
            if pred:
                with _MUTE:
                    out[i] = float(np.mean(evaluate_answer(
                        pred, ans[i] if ans else gold, ds, "qa",
                        subtask=sub[i] if sub else None)))
        return out
    finally:
        os.chdir(cwd)


def paired(a, b, n=6000, seed=0):
    """配对 bootstrap：→ (均值差×100, lo, hi, n_common)。"""
    c = sorted(set(a) & set(b))
    d = (np.array([a[j] for j in c]) - np.array([b[j] for j in c])) * 100
    rg = np.random.default_rng(seed)
    bs = np.array([d[rg.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(c)


if __name__ == "__main__":
    ds, tag, r = sys.argv[1], sys.argv[2], float(sys.argv[3])
    o = read_scores(ds, tag, r)
    print(f"{ds} {tag} ρ={r}: n={len(o)}  均值 {np.mean(list(o.values()))*100:.2f}")
