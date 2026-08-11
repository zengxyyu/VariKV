"""scbench_kv 上残差读出评测的配对显著性检验（2026-08-10 夜跑）。

为什么不能直接读 results.parse 的相对行：每次运行按**自己**的 full-cache
归一化，而 ratio=1.0 那一档记忆不参与、各运行仍差 66.80~70.40（bf16/GPU
非确定性）。除以不同分母会凭空造出几个点的差。这里一律用绝对分 + 按样本配对。

逐样本打分完全照抄 results/parse.py 的主循环（含 preds[1.0] 的补齐条件），
并对基线自检：若逐样本均值与 parse 打印的绝对分不一致就直接报错。
"""
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

DATA = "scbench_kv"
TASK = "qa"
_MUTE = contextlib.redirect_stdout(io.StringIO())
with _MUTE:
    ANSW, SUBT = parse_answer(DATA)


def per_sample(model_tag, ratios):
    """返回 {ratio: [每样本均分]}，含 1.0 参照；照抄 parse.py 的解析逻辑。"""
    rs = list(ratios) + [1.0]
    out = {r: [] for r in rs}
    i = 0
    while True:
        f = f"./results/{DATA}/{i}_{model_tag}/output-{'pair'}.json"
        if not os.path.exists(f):
            break
        d = json.load(open(f))
        preds = {r: [] for r in rs}
        answers = []
        for fmt in [k for k in d if k.startswith(TASK)]:
            for info, text in d[fmt]:
                if info[0] in preds:
                    preds[info[0]].append(text["pruned"])
            if len(preds[1.0]) < len(preds[ratios[-1]]):
                preds[1.0].append(text["full__"])
            answers.append(text["answer"])
        gold = ANSW[i] if ANSW else answers
        sub = SUBT[i] if SUBT else None
        for r in rs:
            if preds[r]:
                with contextlib.redirect_stdout(io.StringIO()):
                    sc = evaluate_answer(preds[r], gold, DATA, TASK, subtask=sub)
                out[r].append(float(np.mean(sc)))
        i += 1
    return out, i


def boot(a, b, n=10000, seed=0):
    """配对 bootstrap：b - a 的均值与 95% CI。"""
    a, b = np.asarray(a), np.asarray(b)
    assert len(a) == len(b), f"样本数不一致 {len(a)} vs {len(b)}"
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d.mean(), lo, hi, (lo > 0) or (hi < 0)


STD = [0.75, 0.5, 0.4, 0.3, 0.2]
LOW = [0.1, 0.05]
M = "qwen2.5-7b-instruct-1m_"

# 自检：parse 打印的绝对分（上面手工跑过一遍），用来验证本脚本的解析一致
EXPECT = {
    M + "fastkvzip__rb_chunk16k_w4096":
        {1.0: 68.20, 0.75: 68.80, 0.5: 71.60, 0.4: 66.40, 0.3: 65.40, 0.2: 45.20},
    M + "fastkvzip_kvlb_chunk16k_w4096":
        {1.0: 68.20, 0.1: 32.60, 0.05: 2.00},
}

GROUPS = [
    ("标准区间 0.75→0.2（基线 rb，门开的一对）", STD,
     M + "fastkvzip__rb_chunk16k_w4096", [
         ("[lm目标] dist 门开0.186", M + "fastkvzip_kvres_chunk16k_w4096_varikvdist16_res"),
         ("[lm目标] point 门开0.287", M + "fastkvzip_kvres_chunk16k_w4096_varikvpoint16_res"),
     ]),
    # 2026-08-11 补跑：gap 目标三档在标准区间（论文 Figure 11 的 Retr.KV 那一格
    # 横轴就是这一段）。基线复用 rb（同 100 条、同配置，不重跑）。
    ("标准区间 0.75→0.2（gap 目标三档，2026-08-11 补跑）", STD,
     M + "fastkvzip__rb_chunk16k_w4096", [
         ("gapf dist 门0.032", M + "fastkvzip_gfsd_chunk16k_w4096_varikvdist16_res"),
         ("gapr dist 门0.014", M + "fastkvzip_grsd_chunk16k_w4096_varikvdist16_res"),
         ("gapr point 门0.024", M + "fastkvzip_grsp_chunk16k_w4096_varikvpoint16_res"),
     ]),
    ("低比例 0.1 / 0.05（全档位）", LOW,
     M + "fastkvzip_kvlb_chunk16k_w4096", [
         ("[lm目标] dist 门开0.186", M + "fastkvzip_kvlres_chunk16k_w4096_varikvdist16_res"),
         ("[lm目标] point 门开0.287", M + "fastkvzip_kvlres_chunk16k_w4096_varikvpoint16_res"),
         ("gapf dist 门0.032", M + "fastkvzip_kvlgf_chunk16k_w4096_varikvdist16_res"),
         ("gapr dist 门0.014", M + "fastkvzip_kvlgr_chunk16k_w4096_varikvdist16_res"),
         ("gapr point 门0.024", M + "fastkvzip_kvlgr_chunk16k_w4096_varikvpoint16_res"),
     ]),
]

for title, ratios, base_tag, variants in GROUPS:
    print("=" * 82)
    print(title)
    base, nb = per_sample(base_tag, ratios)
    for r, want in EXPECT.get(base_tag, {}).items():
        got = np.mean(base[r]) * 100
        assert abs(got - want) < 0.05, f"自检失败 {base_tag} ratio {r}: {got:.2f} != {want}"
    print(f"基线 {base_tag.split('_fastkvzip')[-1]}  n={nb}  "
          f"full-cache {np.mean(base[1.0])*100:.2f}  （自检通过）")
    for name, tag in variants:
        v, nv = per_sample(tag, ratios)
        if nv == 0:
            print(f"  {name:26s} 无结果目录")
            continue
        print(f"  {name:26s} n={nv}  full-cache {np.mean(v[1.0])*100:.2f}")
        for r in ratios:
            if not v[r] or not base[r]:
                continue
            m, lo, hi, sep = boot(base[r], v[r])
            print(f"      ratio {r:<5} 基线 {np.mean(base[r])*100:6.2f}  "
                  f"本档 {np.mean(v[r])*100:6.2f}  Δ {m*100:+7.2f} "
                  f"[{lo*100:+7.2f},{hi*100:+7.2f}] {'★分离' if sep else '未分离'}")
