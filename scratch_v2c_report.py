#!/usr/bin/env python3
"""干净版 v2（`varikv_v2.py`，3 种子）在 11 panel × 9 ratio 上的配对报表。

口径：只报配对 Δ（对同一批 `__g8base`），逐样本 bootstrap，★ = 95% CI 不含 0。
逐种子分别算，再报跨种子均值与散布 —— 一次训练不是一次测量。

**ratio 0.02 是阴性对照不是结果**：`ratio×clen < window` 时 `chunk_ratio` 被置 0，
保留集变成"最后 ratio×clen 个 token"，门控分数完全不参与 ⇒ 改分数恒为 no-op。
门槛 `clen > 4096/ratio`，0.02 需要 204,800 而最长的 scbench_kv 只有 169,428
⇒ **11 个 panel 全部退化**，那一列应当恒为 0；非零就说明实现坏了。
同理 0.05 卡掉 repoqa(72k) 及更短的，0.1 卡掉 many_shot(26k)，gsm/squad 除 1.0 外全退化。
"""
import contextlib, io, json, os, sys
import numpy as np
_P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external/FastKVzip/prefill")
sys.path.insert(0, _P); os.chdir(_P)
from results.parse import parse_answer, evaluate_answer            # noqa: E402
_M = contextlib.redirect_stdout(io.StringIO())
_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch_ctrl_logs")


def DONE(data, seed):
    f = os.path.join(_LOG, f"v2cbench_{data}_s{seed}.log")
    try:
        return "Finished." in open(f, errors="ignore").read()[-4000:]
    except OSError:
        return False

PANEL = {"scbench_kv":"Retr.KV","scbench_prefix_suffix":"Retr.PrefSuf",
         "scbench_repoqa":"Code.RepoQA","squad":"SQuAD","gsm":"GSM8K",
         "scbench_qa_eng":"En.QA","scbench_choice_eng":"En.MultiChoice",
         "scbench_summary":"En.Summary","scbench_vt":"Retr.MultiHop",
         "scbench_mf":"Math.Find","scbench_many_shot":"ICL.ManyShot"}
TOK = {"gsm":86,"squad":203,"scbench_many_shot":26474,"scbench_repoqa":72499,
       "scbench_prefix_suffix":112635,"scbench_summary":117806,
       "scbench_choice_eng":119299,"scbench_qa_eng":122101,"scbench_vt":124551,
       "scbench_mf":149860,"scbench_kv":169428}
RAT = [0.75,0.5,0.4,0.3,0.2,0.1,0.05,0.02]
W = 4096

def per_sample(data, sfx, ratio):
    with _M: ANSW, SUBT = parse_answer(data)
    out, i = {}, 0
    while True:
        f = f"results/{data}/{i}_qwen2.5-7b-instruct-1m_fastkvzip{sfx}/output-pair.json"
        if not os.path.exists(f): break
        dd = json.load(open(f)); p, ans = [], []
        for fmt in [k for k in dd if k.startswith("qa")]:
            for info, txt in dd[fmt]:
                if abs(float(info[0]) - ratio) < 1e-9:
                    p.append(txt["pruned"]); ans.append(txt["answer"])
        if p:
            with _M:
                out[i] = float(np.mean(evaluate_answer(p, ANSW[i] if ANSW else ans,
                                       data, "qa", subtask=SUBT[i] if SUBT else None)))
        i += 1
    return out

def boot(d, n=3000, seed=0):
    r = np.random.default_rng(seed)
    b = np.array([d[r.integers(0,len(d),len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(b,2.5), np.percentile(b,97.5)

W1 = 15
print(f"{'panel':<15}{'full':>6} " + "".join(f"{('ρ=%g'%r):>{W1}}" for r in RAT))
print("-"*(21+W1*len(RAT)))
agg = {r: [] for r in RAT}
for d, name in PANEL.items():
    base = {r: per_sample(d, "__g8base_chunk16k_w4096", r) for r in RAT[:-1]}
    base[0.02] = per_sample(d, "__b002_chunk16k_w4096", 0.02)
    full = per_sample(d, "__g8base_chunk16k_w4096", 1.0)
    line = f"{name:<15}{(np.mean(list(full.values()))*100 if full else float('nan')):>6.1f} "
    for r in RAT:
        ds = []
        for S in (0,1,2):
            A = per_sample(d, f"__v2c_s{S}_chunk16k_w4096_ctrlmmemo8", r)
            B = base.get(r) or {}
            c = sorted(set(A) & set(B))
            # **完成判定看日志的 `Finished.`，不看条数** —— choice_eng 只有 18 条、
            # qa_eng 20、many_shot 54、summary 70、vt 90，按 `len(c) >= 100` 过滤会
            # 把这些**完整**的 panel 当成截断的丢掉（首版就这么丢了 5 个 panel）。
            if not DONE(d, S):
                continue
            v = (np.array([A[j] for j in c]) - np.array([B[j] for j in c]))*100
            ds.append(boot(v))
        deg = "" if (r >= 1.0 or TOK[d] > W/r) else "°"   # ° = 该格结构性退化
        if not ds:
            line += f"{'—':>{W1}}"
        else:
            m = np.mean([x[0] for x in ds])
            sig = all(x[1] > 0 for x in ds) or all(x[2] < 0 for x in ds)
            agg[r].append(m)
            line += f"{('%+.2f%s%s(%d)'%(m,'★' if sig else '',deg,len(ds))):>{W1}}"
    print(line)
print("-"*(21+W1*len(RAT)))
line = f"{'**均值**':<15}{'':>6} "
for r in RAT:
    v = agg[r]
    line += f"{(('%+.2f (%d)'%(np.mean(v),len(v))) if v else '—'):>{W1}}"
print(line)
print("\n★ = 三个种子的 95% CI 同向且都不含 0；括号内 = 参与平均的种子数")
print("° = 该 (panel,ratio) 结构性退化（ratio×clen < window ⇒ 只留最近的 token，"
      "门控分数不参与），Δ 应恒为 0")
