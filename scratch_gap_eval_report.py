"""gap ckpt 评测的配对显著性检验。

为什么必须做：many_shot 基线自身就非单调（0.75→96.08 而 0.5→100.00），
指标本身的抖动量级和我们要报的差值同量级。此前 prefix_suffix 的教训是
——单数据集上的大间隔可以完全是一个数据集的伪影。

按**样本**配对（两次运行跑的是同一批 54 条 context，同序），
对每样本 5 个 query 的均值做配对 bootstrap。
"""
import contextlib
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                + "/external/FastKVzip/prefill")
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/external/FastKVzip/prefill")

from results.parse import parse_answer                      # noqa: E402
from results.metric import evaluate_answer                  # noqa: E402

DATA = "scbench_many_shot"
# evaluate_answer / parse_answer 会往 stdout 刷大量 include_score_manyshot，吞掉。
_MUTE = contextlib.redirect_stdout(io.StringIO())
with _MUTE:
    ANSW, SUBT = parse_answer(DATA)


def per_sample(model_tag, ratios):
    """返回 {ratio: [每样本均分]}，外加 full-cache 参照。"""
    out = {r: [] for r in list(ratios) + [1.0]}
    i = 0
    while True:
        f = f"./results/{DATA}/{i}_{model_tag}/output-pair.json"
        if not os.path.exists(f):
            break
        d = json.load(open(f))
        preds, answers = {r: [] for r in list(ratios) + [1.0]}, []
        for fmt in [k for k in d if k.startswith("qa")]:
            for info, text in d[fmt]:
                if info[0] in preds:
                    preds[info[0]].append(text["pruned"])
            preds[1.0].append(text["full__"])
            answers.append(text["answer"])
        gold = ANSW[i] if ANSW else answers
        for r in preds:
            if preds[r]:
                with contextlib.redirect_stdout(io.StringIO()):
                    sc = evaluate_answer(preds[r], gold, DATA, "qa", subtask=None)
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

GROUPS = [
    ("标准区间 0.75→0.2", STD, M + "fastkvzip__ret_chunk16k_w4096", [
        ("gapf dist (固定0.3)", M + "fastkvzip_gapf_chunk16k_w4096_varikvdist16_res"),
        ("gapr dist (随机比例)", M + "fastkvzip_gapr_chunk16k_w4096_varikvdist16_res"),
        ("gapr point (随机比例)", M + "fastkvzip_gapr_chunk16k_w4096_varikvpoint16_res"),
        # lm 目标训出的残差 ckpt（ckpt_stage2b_res），门是开的 sigmoid≈0.186，
        # 与上面三个 gap 目标（门≈0.014~0.032，等于关掉）形成直接对照。
        ("[lm目标] rt dist 门开", M + "fastkvzip__rt_chunk16k_w4096_varikvdist16_res"),
        ("[lm目标] rt point 门开", M + "fastkvzip__rt_chunk16k_w4096_varikvpoint16_res"),
    ]),
    ("低比例 0.1 / 0.05", LOW, M + "fastkvzip__low_chunk16k_w4096", [
        ("gapf dist (固定0.3)", M + "fastkvzip_gapfl_chunk16k_w4096_varikvdist16_res"),
        ("gapr dist (随机比例)", M + "fastkvzip_gaprl_chunk16k_w4096_varikvdist16_res"),
        ("gapr point (随机比例)", M + "fastkvzip_gaprl_chunk16k_w4096_varikvpoint16_res"),
        ("[旧] dist KV注入", M + "fastkvzip__low_chunk16k_w4096_varikvdist16"),
        ("[旧] 读出置零 rozero",
         M + "fastkvzip__low_chunk16k_w4096_varikvdist16_rozero"),
    ]),
]

for title, ratios, base_tag, variants in GROUPS:
    print("=" * 78)
    print(title)
    base, nb = per_sample(base_tag, ratios)
    print(f"基线 {base_tag.split('_fastkvzip')[-1]}  n={nb} 样本  "
          f"full-cache 绝对分 {np.mean(base[1.0])*100:.2f}")
    for name, tag in variants:
        v, nv = per_sample(tag, ratios)
        if nv == 0:
            print(f"  {name:24s} 无结果目录")
            continue
        print(f"  {name:24s} n={nv}  full-cache {np.mean(v[1.0])*100:.2f}")
        for r in ratios:
            if not v[r] or not base[r]:
                continue
            m, lo, hi, sep = boot(base[r], v[r])
            print(f"      ratio {r:<5} 基线 {np.mean(base[r])*100:6.2f}  "
                  f"本档 {np.mean(v[r])*100:6.2f}  Δ {m*100:+6.2f} "
                  f"[{lo*100:+6.2f},{hi*100:+6.2f}] {'★分离' if sep else '未分离'}")
