#!/usr/bin/env python3
"""全网格报告：3 臂 × 11 panel × 8 ratio，配对 bootstrap。

绝对分只在两臂样本集**完全一致**时才可比（CLAUDE.md：慢的那一臂没跑到的样本会
被交集丢掉，绝对基线因此漂移，而配对 Δ 稳定）。所以这里一律报配对 Δ 并附 n。
"""
import numpy as np, importlib.util as iu, os, sys, json, contextlib, io
_P = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external/FastKVzip/prefill')
sys.path.insert(0, _P); os.chdir(_P)     # chdir 之后再 import，否则 results 包找不到
from results.parse import parse_answer, evaluate_answer
RAT = [1.0, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
PANEL = {"scbench_kv":"Retr.KV","scbench_vt":"Retr.MultiHop","scbench_prefix_suffix":"Retr.PrefSuf",
         "scbench_repoqa":"Code.RepoQA","gsm":"GSM8K","squad":"SQuAD","scbench_qa_eng":"En.QA",
         "scbench_choice_eng":"En.MultiChoice","scbench_summary":"En.Summary",
         "scbench_mf":"Math.Find","scbench_many_shot":"ICL.ManyShot"}
_M = contextlib.redirect_stdout(io.StringIO())

def per_sample(data, suffix, ratio):
    with _M: ANSW, SUBT = parse_answer(data)
    out, i = {}, 0
    while True:
        f = f"results/{data}/{i}_qwen2.5-7b-instruct-1m_fastkvzip{suffix}/output-pair.json"
        if not os.path.exists(f): break
        dd = json.load(open(f)); p, ans = [], []
        for fmt in [k for k in dd if k.startswith("qa")]:
            for info, txt in dd[fmt]:
                if abs(float(info[0]) - ratio) < 1e-9: p.append(txt["pruned"]); ans.append(txt["answer"])
        gold = ANSW[i] if ANSW else ans; sub = SUBT[i] if SUBT else None
        if p:
            with _M: out[i] = float(np.mean(evaluate_answer(p, gold, data, "qa", subtask=sub)))
        i += 1
    return out

def boot(d, n=4000, seed=0):
    r = np.random.default_rng(seed)
    b = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

hdr = f"{'panel':<15}{'n':>4}{'full':>7}" + "".join(f"{r:>16}" for r in RAT[1:])
print(hdr); print("-" * len(hdr))
agg = {r: {"v2": [], "v3": []} for r in RAT}
for d in PANEL:
    # **双下划线**：`--tag _g8xxx` 的前导下划线会在目录名里变成 `fastkvzip__g8xxx`
    P = {a: {r: per_sample(d, f"__g8{a}_chunk16k_w4096"
                           + ("" if a == "base" else "_ctrlmmemo8"), r)
             for r in RAT} for a in ("base", "v2", "v3")}
    if not P["base"][0.1]: continue
    for a in ("v2","v3"):
        if not P[a][0.1]: continue
        row = f"{PANEL[d] if a=='v2' else '':<15}"
        c0 = sorted(set(P["base"][1.0]) & set(P[a][1.0]))
        row += f"{len(c0):>4}" + f"{np.mean([P['base'][1.0][j] for j in c0])*100:>7.1f}" if c0 else f"{0:>4}{0:>7.1f}"
        for r in RAT[1:]:
            c = sorted(set(P["base"][r]) & set(P[a][r]))
            if not c: row += f"{'-':>16}"; continue
            dd = (np.array([P[a][r][j] for j in c]) - np.array([P["base"][r][j] for j in c]))*100
            m, lo, hi = boot(dd); agg[r][a].append(m)
            row += f"{m:>+9.2f}{'★' if (lo>0 or hi<0) else ' '}{a:>6}"
        print(row)
print("-" * len(hdr))
for a in ("v2","v3"):
    print(f"{'均值 '+a:<15}{'':>4}{'':>7}" + "".join(
        f"{np.mean(agg[r][a]):>+11.2f}({len(agg[r][a]):>2})" if agg[r][a] else f"{'-':>16}" for r in RAT[1:]))
