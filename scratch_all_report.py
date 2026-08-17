#!/usr/bin/env python3
"""把学习残差（v2/v3）与训练无关质心（K=16/K=1024）放进同一张表。

统一口径，三条都是这个项目付过学费的：

1. **只报配对 Δ，不报跨运行的绝对分。** `results.parse` 的相对行按各自的满缓存分
   归一，而满缓存分逐运行漂移；绝对分也只在两臂样本集完全一致时才可比（慢的那一臂
   没跑到的样本会被交集丢掉）。
2. **共同基线**用 `__g8base`（全 11 panel × 8 ratio 的那次），这样质心与残差对的是
   同一批基线数字，两条线之间才可比。
3. **完成判定看日志的 `Finished.`，不看结果文件计数** —— choice_eng 18 条、
   qa_eng 20、many_shot 54、repoqa 88、vt 90，计数法会把完整的当成截断的。

`--md` 输出 markdown。
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

_P = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "external/FastKVzip/prefill")
sys.path.insert(0, _P)
os.chdir(_P)
from results.parse import parse_answer, evaluate_answer          # noqa: E402

RAT = [1.0, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]
# 基线在 0.02 上是另一批（`__b002`），因为 `__g8base` 只跑了 8 个 ratio
BASE_SFX = {0.02: "__b002_chunk16k_w4096"}
_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch_ctrl_logs")
# `ratio×clen < window` ⇒ chunk_ratio 置 0、只留最近的 token、门控分数不参与
# ⇒ 任何改分数的方法恒为 no-op。这些格标 `°`，Δ 应为 0，非零即实现有问题。
TOK = {"gsm": 86, "squad": 203, "scbench_many_shot": 26474, "scbench_repoqa": 72499,
       "scbench_prefix_suffix": 112635, "scbench_summary": 117806,
       "scbench_choice_eng": 119299, "scbench_qa_eng": 122101,
       "scbench_vt": 124551, "scbench_mf": 149860, "scbench_kv": 169428}


def v2c_done(data, seed):
    """**完成判定看日志的 `Finished.`，不看条数。** choice_eng 只有 18 条、
    qa_eng 20、many_shot 54、summary 70、vt 90 —— 按条数过滤会把这些**完整**的
    panel 当成截断的丢掉（本项目已犯过两次）。"""
    f = os.path.join(_LOG, f"v2cbench_{data}_s{seed}.log")
    try:
        return "Finished." in open(f, errors="ignore").read()[-4000:]
    except OSError:
        return False
PANEL = {"scbench_kv": "Retr.KV", "scbench_prefix_suffix": "Retr.PrefSuf",
         "scbench_repoqa": "Code.RepoQA", "squad": "SQuAD", "gsm": "GSM8K",
         "scbench_qa_eng": "En.QA", "scbench_choice_eng": "En.MultiChoice",
         "scbench_summary": "En.Summary", "scbench_vt": "Retr.MultiHop",
         "scbench_mf": "Math.Find", "scbench_many_shot": "ICL.ManyShot"}
# 质心的 tag 是分几批跑出来的，命名不统一，这里显式列出而不是拼规则——
# 拼规则拼错会静默返回空字典，然后整格显示成"缺数据"，很难发现。
# 目录名的形状是 `fastkvzip_<tag>_chunk16k_w4096_cen<K>` —— tag 与 `_cen<K>` 之间
# 还夹着 `_chunk16k_w4096`。首版漏了中间那段，四列全空且不报错（`per_sample` 找不到
# 目录就返回空字典），只能靠"整片都是 —"发现。
_C = "_chunk16k_w4096_cen"
CEN = {
    "scbench_kv": {16: f"__cen16{_C}16", 1024: f"__cen1024{_C}1024"},
    "scbench_vt": {16: f"__p2_cen16_vt{_C}16", 1024: f"__p2_cen1024_vt{_C}1024"},
    "scbench_prefix_suffix": {16: f"__p2_cen16_ps{_C}16",
                              1024: f"__p2_cen1024_ps{_C}1024"},
    "scbench_repoqa": {16: f"__p2_cen16_rq{_C}16", 1024: f"__p2_cen1024_rq{_C}1024"},
}
for d in ("gsm", "squad", "scbench_qa_eng", "scbench_choice_eng",
          "scbench_summary", "scbench_mf", "scbench_many_shot"):
    CEN[d] = {16: f"__cen16_{d}{_C}16", 1024: f"__cen1024_{d}{_C}1024"}
# 另外两批：`_c23*` 覆盖 ratio 0.3/0.2（6 panel），`_r05c*` 覆盖 0.05/0.1（5 panel）。
# 同一 (panel, K) 的不同 ratio 散在不同 tag 里，所以取值时要按 ratio 找对应那批。
CEN23 = {d: {16: f"__c2316_{d}{_C}16", 1024: f"__c231024_{d}{_C}1024"}
         for d in ("scbench_kv", "scbench_mf", "scbench_prefix_suffix",
                   "scbench_repoqa", "scbench_summary", "scbench_vt")}
CEN05 = {d: {16: f"__r05c16{_C}16", 1024: f"__r05c1024{_C}1024"}
         for d in ("scbench_kv", "scbench_prefix_suffix", "scbench_repoqa",
                   "scbench_summary", "scbench_vt")}


def cen_sfx(d, K, r):
    """同一 (panel,K) 的不同 ratio 分散在三批 tag 里，按 ratio 选对应那批。"""
    if r in (0.3, 0.2) and d in CEN23:
        return CEN23[d][K]
    if r == 0.05 and d in CEN05:
        return CEN05[d][K]
    return CEN[d][K] if r == 0.1 else None

_M = contextlib.redirect_stdout(io.StringIO())


def per_sample(data, suffix, ratio):
    with _M:
        ANSW, SUBT = parse_answer(data)
    out, i = {}, 0
    while True:
        f = f"results/{data}/{i}_qwen2.5-7b-instruct-1m_fastkvzip{suffix}/output-pair.json"
        if not os.path.exists(f):
            break
        dd = json.load(open(f))
        p, ans = [], []
        for fmt in [k for k in dd if k.startswith("qa")]:
            for info, txt in dd[fmt]:
                if abs(float(info[0]) - ratio) < 1e-9:
                    p.append(txt["pruned"]); ans.append(txt["answer"])
        gold = ANSW[i] if ANSW else ans
        sub = SUBT[i] if SUBT else None
        if p:
            with _M:
                out[i] = float(np.mean(evaluate_answer(p, gold, data, "qa",
                                                       subtask=sub)))
        i += 1
    return out


def boot(d, n=4000, seed=0):
    r = np.random.default_rng(seed)
    b = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def cell(base, arm):
    c = sorted(set(base) & set(arm))
    if not c:
        return None
    d = (np.array([arm[j] for j in c]) - np.array([base[j] for j in c])) * 100
    m, lo, hi = boot(d)
    return m, (lo > 0 or hi < 0), len(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    ARMS = [("v2", lambda d: "__g8v2_chunk16k_w4096_ctrlmmemo8"),
            ("v2c", "SEEDS"),        # 干净版 v2（varikv_v2.py），3 种子逐个算再平均
            ("v3", lambda d: "__g8v3_chunk16k_w4096_ctrlmmemo8"),
            ("cen16", None), ("cen1024", None)]     # 质心按 ratio 选 tag
    agg = {a_: {r: [] for r in RAT} for a_, _ in ARMS}
    rows = []
    for d, name in PANEL.items():
        B = {r: per_sample(d, BASE_SFX.get(r, "__g8base_chunk16k_w4096"), r)
             for r in RAT}
        full = np.mean(list(B[1.0].values())) * 100 if B[1.0] else float("nan")
        for a_, sfx in ARMS:
            seeds = None
            try:
                if sfx == "SEEDS":                       # 干净版 v2：3 个训练种子
                    seeds = [S for S in (0, 1, 2) if v2c_done(d, S)]
                    A = {}
                elif sfx is None:                        # 质心：逐 ratio 找 tag
                    K = 16 if a_ == "cen16" else 1024
                    A = {}
                    for r in RAT:
                        t = cen_sfx(d, K, r)
                        if t:
                            A[r] = per_sample(d, t, r)
                else:
                    A = {r: per_sample(d, sfx(d), r) for r in RAT}
            except Exception:
                A, seeds = {}, None
            got = {}
            for r in RAT[1:]:
                if not B.get(r):
                    continue
                if seeds is not None:                    # 多种子：逐种子算再平均
                    ms = []
                    for S in seeds:
                        c = cell(B[r], per_sample(
                            d, f"__v2c_s{S}_chunk16k_w4096_ctrlmmemo8", r))
                        if c:
                            ms.append(c)
                    if not ms:
                        continue
                    m_ = float(np.mean([x[0] for x in ms]))
                    sig = all(x[1] for x in ms)          # 全部种子都显著才给 ★
                    sd_ = float(np.std([x[0] for x in ms])) if len(ms) > 1 else None
                    got[r] = (m_, sig, len(ms), sd_)
                    agg[a_][r].append(m_)
                    continue
                if not A.get(r):
                    continue
                c = cell(B[r], A[r])
                if c:
                    got[r] = (c[0], c[1], 1)      # 单 ckpt ⇒ 种子数 1
                    agg[a_][r].append(c[0])
            rows.append((name, full, a_, got, d))
    W = 15
    hd = f"| {'panel':<15}| {'full':>5} | {'arm':<8}|" + "".join(
        f" {('ρ=%g' % r):>{W}} |" for r in RAT[1:])
    print(hd)
    print("|" + "|".join(["-" * 16, "-" * 7, "-" * 9]
                         + ["-" * (W + 2)] * (len(RAT) - 1)) + "|")
    last = None
    for name, full, a_, got, name_d in rows:
        n = name if name != last else ""
        last = name
        line = f"| {n:<15}| {full:>5.1f} | {a_:<8}|"
        for r in RAT[1:]:
            if r not in got:
                line += f" {'—':>{W}} |"
            else:
                m, sig, ns = got[r][0], got[r][1], got[r][2]
                sd = got[r][3] if len(got[r]) > 3 else None
                deg = "" if (r >= 1.0 or TOK.get(name_d, 10**9) > 4096 / r) else "°"
                # **所有臂都标种子数**：v2/v3/质心那几行全是 n=1（表里 v2 用的是
                # 单个 ckpt `ctrl_b_a1_s0`；`+4.27 ± 0.19` 的三种子数字只存在于
                # scbench_kv @0.1 那一格，从没有 11×8 的三种子版本）。只给 v2c 标
                # 会让人误以为别人是多种子的。
                # **只有多种子的格子标 (n) 与散布**：单种子是常态（表头已声明），
                # 每格都印 `(1)` 只是噪声。n≥2 时印 `±跨种子标准差(n)`。
                body = "%+.2f" % m if ns < 2 else "%+.2f±%.2f" % (m, sd or 0.0)
                tail = ("★" if sig else "") + deg + (f"({ns})" if ns >= 2 else "")
                line += f" {body + tail:>{W}} |"
        print(line)
    print("|" + "|".join(["-" * 16, "-" * 7, "-" * 9]
                         + ["-" * (W + 2)] * (len(RAT) - 1)) + "|")
    for a_, _ in ARMS:
        line = f"| {'**均值**':<15}| {'':>5} | {a_:<8}|"
        for r in RAT[1:]:
            v = agg[a_][r]
            line += f" {(('%+.2f (%d)' % (np.mean(v), len(v))) if v else '—'):>{W}} |"
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())
