#!/usr/bin/env python3
"""质心 @ ratio 0.05（外加 0.1）× 5 个代表性面板，逐样本配对 bootstrap，绝对分。

口径（沿用本仓库既有纪律）：
- **只报绝对分**，绝不读 `results.parse` 的相对行——每个 run 用自己的满缓存分做分母，
  跨 run 不可比。
- **逐样本配对**，三臂取共同样本交集；样本数不齐时截到交集并把丢弃数打出来。
- 满缓存参考 `full__` 从**同一批 run 自己的文件**里取。质心在 ratio 1.0 时没有任何东西被
  驱逐 ⇒ 摘要为空 ⇒ λ=1 ⇒ 不注入，所以三臂的 `full__` 应当完全一致；脚本把这件事**当断言检查**
  （learned-memory 那轮就是栽在「空记忆仍然注入、污染了满缓存参考」上）。
- ratio 0.1 同批带出来，用来和既有的独立 run（tag `_cen16` / `_b01`）对账。
"""
import contextlib
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
os.chdir(os.path.join(ROOT, "external/FastKVzip/prefill"))
from results.parse import evaluate_answer, parse_answer          # noqa: E402

MODEL = "qwen2.5-7b-instruct-1m"
_M = contextlib.redirect_stdout(io.StringIO())
PANEL = {"scbench_kv": "Retr.KV", "scbench_repoqa": "Code.RepoQA",
         "scbench_prefix_suffix": "Retr.Prefix-Suffix", "scbench_vt": "Retr.MultiHop",
         "scbench_summary": "En.Summary"}
ARMS = [("基线", "__r05b_chunk16k_w4096"),
        ("K=16", "__r05c16_chunk16k_w4096_cen16"),
        ("K=1024", "__r05c1024_chunk16k_w4096_cen1024")]


def per_sample(data, suffix, ratio, task="qa"):
    """→ ({i: pruned 分}, {i: 满缓存分})，逐样本。"""
    with _M:
        ANSW, SUBT = parse_answer(data)
    pr, fu, i = {}, {}, 0
    while True:
        f = f"./results/{data}/{i}_{MODEL}_fastkvzip{suffix}/output-pair.json"
        if not os.path.exists(f):
            break
        dd = json.load(open(f))
        p, q, ans = [], [], []
        for fmt in [k for k in dd if k.startswith(task)]:
            for info, text in dd[fmt]:
                if abs(float(info[0]) - ratio) < 1e-9:
                    p.append(text["pruned"]); q.append(text["full__"]); ans.append(text["answer"])
        gold = ANSW[i] if ANSW else ans
        sub = SUBT[i] if SUBT else None
        if p:
            with _M:
                pr[i] = float(np.mean(evaluate_answer(p, gold, data, task, subtask=sub)))
                fu[i] = float(np.mean(evaluate_answer(q, gold, data, task, subtask=sub)))
        i += 1
    return pr, fu


def boot(d, n=10000, seed=0):
    r = np.random.default_rng(seed)
    d = np.asarray(d)
    s = d[r.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), float(np.quantile(s, .025)), float(np.quantile(s, .975))


def main():
    for ratio in (0.05, 0.1):
        print(f"\n{'='*118}")
        print(f"【训练无关质心 @ ratio {ratio}】绝对分，逐样本配对 bootstrap，★=95%CI 不含 0")
        print("-" * 118)
        print(f"{'论文面板':<22}{'n':>5}{'满缓存':>8}{'基线':>8}{'headroom':>10}"
              f"{'K=16 Δ':>22}{'K=1024 Δ':>22}")
        for data in PANEL:
            got = {}
            for label, suf in ARMS:
                pr, fu = per_sample(data, suf, ratio)
                if pr:
                    got[label] = (pr, fu)
            if len(got) < 3:
                print(f"{PANEL[data]:<22} 缺臂：只有 {sorted(got) or '无'}")
                continue
            common = sorted(set.intersection(*[set(v[0]) for v in got.values()]))
            if not common:
                print(f"{PANEL[data]:<22} 无共同样本")
                continue
            # 断言：三臂的满缓存参考应当一致（质心在 ratio 1.0 不注入）
            fulls = [np.mean([got[l][1][i] for i in common]) * 100 for l, _ in ARMS]
            if max(fulls) - min(fulls) > 1e-6:
                print(f"  ⚠ {PANEL[data]}：三臂满缓存参考不一致 {['%.2f'%f for f in fulls]}"
                      f" —— 说明质心在 ratio 1.0 也在注入，满缓存参考被污染，"
                      f"下面这一行的 headroom 不可信")
            b = np.array([got["基线"][0][i] for i in common]) * 100
            full = fulls[0]
            n_drop = max(len(v[0]) for v in got.values()) - len(common)
            row = (f"{PANEL[data]:<22}{len(common):>5}{full:>8.2f}{b.mean():>8.2f}"
                   f"{full-b.mean():>+10.2f}")
            for label in ("K=16", "K=1024"):
                c = np.array([got[label][0][i] for i in common]) * 100
                mm, lo, hi = boot(c - b)
                row += f"{mm:>+9.2f}[{lo:>+6.1f},{hi:>+6.1f}]{'★' if (lo > 0 or hi < 0) else ' '}"
            print(row + (f"   （丢弃 {n_drop} 条不齐样本）" if n_drop else ""))
    print(f"\n{'='*118}")
    print("读法：headroom = 满缓存 − 基线，是「完美恢复」能拿回的上限；Δ 为负且 headroom 也为负的面板"
          "\n（Retr.MultiHop）是**预期内**的——那里压缩本身就优于满缓存，忠实恢复必然掉分。")


if __name__ == "__main__":
    sys.exit(main())
