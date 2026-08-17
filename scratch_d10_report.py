#!/usr/bin/env python3
"""v2 档四臂的下游报表 —— +4.27 到底来自校准还是 KV 内容。

口径（都是项目付过学费的）：
0. **只有跑满（≥90% 基线样本）的格子进跨种子聚合**，未跑满的标 `~` 只作参考。
1. 只报**配对 Δ**（对同一批 `__g8base`），逐样本 bootstrap，★ = 95% CI 不含 0。
2. **逐种子分别报，再报跨种子散布** —— 一次训练不是一次测量（v1 同代码三次重训
   跨度 39 分）。
3. 分母是 **v2 自己的三个种子**，与四臂同数据（那 10 篇）、同 8/2 划分、同 320 步、
   同 U^full 教师。不用 v3：v3 的 trace 目录与教师靶子都不同。
   注意 v2 的三个种子只评过 **ratio 0.1**；0.2 只有 `__g8v2` 一个种子。
"""
import contextlib, io, json, os, sys
import numpy as np
_P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external/FastKVzip/prefill")
sys.path.insert(0, _P); os.chdir(_P)
from results.parse import parse_answer, evaluate_answer            # noqa: E402
_M = contextlib.redirect_stdout(io.StringIO())


def _d10_sfx(name, S):
    """**两种拼法都要认。** tag 里的 `ctrlm{mode[:4]}` 曾用覆盖前的 mode 拼：
    修 `--ctrlm_mode` 默认值之前跑的四臂是 `ctrlmstat8`，之后跑的因子臂是
    `ctrlmmemo8`。两批**行为完全相同**（CalibScorer 恒为 memoryless），只是名字
    不同。写死一种会让新的那批整片显示成"缺数据"。"""
    for tag in ("memo", "stat"):
        p = (f"__d10{name}_s{S}_chunk16k_w4096_ctrlm{tag}8_{name}")
        if os.path.exists(f"results/{DATA}/0_qwen2.5-7b-instruct-1m_fastkvzip{p}"):
            return p
    return f"__d10{name}_s{S}_chunk16k_w4096_ctrlmmemo8_{name}"
DATA = "scbench_kv"

def per_sample(sfx, ratio):
    with _M: ANSW, SUBT = parse_answer(DATA)
    out, i = {}, 0
    while True:
        f = f"results/{DATA}/{i}_qwen2.5-7b-instruct-1m_fastkvzip{sfx}/output-pair.json"
        if not os.path.exists(f): break
        dd = json.load(open(f)); p, ans = [], []
        for fmt in [k for k in dd if k.startswith("qa")]:
            for info, txt in dd[fmt]:
                if abs(float(info[0]) - ratio) < 1e-9:
                    p.append(txt["pruned"]); ans.append(txt["answer"])
        if p:
            with _M:
                out[i] = float(np.mean(evaluate_answer(p, ANSW[i] if ANSW else ans,
                                                       DATA, "qa",
                                                       subtask=SUBT[i] if SUBT else None)))
        i += 1
    return out

def boot(d, n=4000, seed=0):
    r = np.random.default_rng(seed)
    b = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

def delta(A, B):
    c = sorted(set(A) & set(B))
    if not c: return None
    return (np.array([A[j] for j in c]) - np.array([B[j] for j in c])) * 100, c

# **臂列表从 CalibScorer.MODES 派生，不手抄第二份。** 手抄过一次的代价：
# scratch_ctrl_train.py 的 --arch choices 漏掉后加的因子臂，12 个训练全在 argparse
# 被拒、日志只留一行 usage，调度器看不出异常、队列照样排空。
import torch                                                        # noqa: E402
from attention.calib_scorer import CalibScorer                      # noqa: E402

_NOTE = {"bias": "纯预算平移", "affine": "位置+尺度",
         "scalar": "z+mg+rs+头嵌入（全局竞争态齐全）",
         "kv": "只看KV，看不到 s⁰", "k": "只看K", "v": "只看V",
         "sz": "只有 z + 头嵌入（无全局态）", "szr": "z+rs（有 σ_g，无 τ）",
         "szm": "z+mg（τ 与 σ_g 都有）", "szmr0": "z+mg+rs，**无头嵌入**"}
# 展示顺序按"信息量递增"排，便于读出单调性；集合本身来自 MODES 并断言全覆盖。
_ORDER = ["bias", "affine", "sz", "szr", "szm", "scalar", "szmr0", "kv"]
_SKIP = {"k", "v"}                       # 未训练，显式跳过而不是默默漏掉
assert set(_ORDER) | _SKIP == set(CalibScorer.MODES), \
    f"MODES 变了但报表没跟上：{set(CalibScorer.MODES) ^ (set(_ORDER) | _SKIP)}"

def _npar(name):
    """参数量从 ckpt 现读，不写死 —— 写死的数字改架构后会静默说谎。"""
    for S in (0, 1, 2):
        f = os.path.join(os.path.dirname(_P), "../..", f"varikv/d10_{name}_s{S}.pt",
                         "memoryless.pt")
        if os.path.exists(f):
            return sum(v.numel() for v in
                       torch.load(f, map_location="cpu")["state"].values())
    return 0

ARMS = [(n, _npar(n), _NOTE[n]) for n in _ORDER if _npar(n)]
V2 = {0.1: {0: "__b2memoryless_chunk16k_w4096_ctrlmmemo8",
            1: "__b2s1me_chunk16k_w4096_ctrlmmemo8",
            2: "__b2s2me_chunk16k_w4096_ctrlmmemo8"},
      0.2: {0: "__g8v2_chunk16k_w4096_ctrlmmemo8"}}

for RATIO in (0.1, 0.2):
    B = per_sample("__g8base_chunk16k_w4096", RATIO)
    print(f"\n{'='*86}\n{DATA} @ ratio {RATIO}　基线 n={len(B)} 绝对分 "
          f"{np.mean(list(B.values()))*100:.2f}\n{'='*86}")
    print(f"{'臂':<10}{'参数':>7}  " + "".join(f"{'s'+str(s):>14}" for s in (0,1,2))
          + f"{'跨种子均±散布':>18}   机制")
    rows = []
    for name, npar, note in ARMS + [("v2 (full)", 637828, "参照")]:
        ds, cell = [], []
        for S in (0, 1, 2):
            sfx = V2[RATIO].get(S) if name.startswith("v2") else \
                  _d10_sfx(name, S)
            A = per_sample(sfx, RATIO) if sfx else {}
            r = delta(A, B) if A else None
            if r is None: cell.append("—"); continue
            d, c = r; m, lo, hi = boot(d)
            # **未跑满的格子不进跨种子聚合，并标 `~`。** 项目付过两次学费：
            # Math.Find 在 38/100 上读 −3.95★，跑满 100 变 −2.33 不显著。这里若不拦，
            # 一个 n=1 的格子会和 n=84 的平均出 `+12.26 ± 7.74` 这种纯噪声。
            full = len(c) >= 0.9 * len(B)
            if full: ds.append(m)
            cell.append(f"{m:+.2f}{'★' if (lo>0 or hi<0) else ''}"
                        f"{'' if full else '~'}({len(c)})")
        agg = f"{np.mean(ds):+.2f} ± {np.std(ds):.2f}" if len(ds) > 1 else \
              (f"{ds[0]:+.2f} (1种子)" if ds else "—")
        print(f"{name:<10}{npar:>7}  " + "".join(f"{x:>14}" for x in cell)
              + f"{agg:>18}   {note}")
        rows.append((name, ds))
    # 臂间 / 对 v2 的种子级配对
    d = dict(rows)
    print(f"\n  种子级配对（各臂三个种子的 Δ 相减，n=3，只看方向与量级）:")
    # 因子臂的关键配对 —— 直接对应 ICLR_PLAN §四之五 的预注册四结局判读表
    for a, b in (("sz","affine"),("szm","sz"),("szr","sz"),
                 ("scalar","szm"),("szmr0","scalar"),
                 ("scalar","bias"),("scalar","affine"),("kv","scalar"),
                 ("scalar","v2 (full)"),("kv","v2 (full)")):
        if len(d.get(a,[]))==3 and len(d.get(b,[]))>=1:
            if len(d[b])==3:
                di = np.array(d[a])-np.array(d[b])
                print(f"    {a:<10} − {b:<10} {di.mean():+7.2f} ± {di.std():.2f}   逐种子 "
                      + " ".join(f"{x:+.2f}" for x in di))
            else:
                print(f"    {a:<10} − {b:<10} {np.mean(d[a])-d[b][0]:+7.2f}   (对照只有 1 个种子)")
