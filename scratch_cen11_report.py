"""免训练质心方法在 Figure-11 全部 11 个面板上的报告（ratio 0.1）。

判据：**绝对分数 + 逐样本配对 bootstrap**，绝不用 `results.parse` 的相对行
（每个 run 用自己的满缓存分数做分母，跨 run 不可比 —— 2026-08-10/11 两次踩过）。

三个必须一起看的东西：
  headroom = 满缓存 − 基线      该面板最多还能捞回多少（可以是负的）
  Δ        = 质心 − 基线        配对，同一批样本
  符号一致  Δ 与 headroom 同号   忠实还原应有的行为：该补的地方补、
                                本来压缩就更好的地方（负 headroom）会掉

**同批基线的 tag 每个面板不一样**（历史原因，不同批次跑的），列在 BASE 里；
质心的 tag 也不统一（`scbench_kv` 没有面板后缀，`vt/rq/ps` 走 `p2_` 前缀）。
一律读逐样本 json、调 harness 自己的 `evaluate_answer` 算绝对分。

用法： .venv/bin/python scratch_cen11_report.py
"""
import contextlib
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
os.chdir(os.path.join(ROOT, "external/FastKVzip/prefill"))
from results.parse import parse_answer, evaluate_answer   # noqa: E402

MODEL, RATIO, LEVEL = "qwen2.5-7b-instruct-1m", 0.1, "pair"

# 论文 Figure-11 面板名 ← 数据集 id（论文用 SCBench 显示名，grep 数据集 id 找不到）
PANEL = {
    "scbench_kv": "Retr.KV", "scbench_prefix_suffix": "Retr.Prefix-Suffix",
    "scbench_repoqa": "Code.RepoQA", "squad": "SQuAD", "gsm": "GSM8K",
    "scbench_qa_eng": "En.QA", "scbench_choice_eng": "En.MultiChoice",
    "scbench_summary": "En.Summary", "scbench_vt": "Retr.MultiHop",
    "scbench_mf": "Math.Find", "scbench_many_shot": "ICL.ManyShot",
}
# 同批基线的 tag（每个面板不同批次跑的，不统一）
BASE = {
    "scbench_kv": "__b01_chunk16k_w4096",
    "scbench_summary": "__fig11_scbench_summary_base_chunk16k_w4096",
    "scbench_mf": "__fig11_scbench_mf_base_chunk16k_w4096",
    "scbench_qa_eng": "__fig11_scbench_qa_eng_base_chunk16k_w4096",
}
BASE_DEFAULT = "__kls_base_chunk16k_w4096"
# 质心臂的 tag（同样不统一）
CEN = {
    "scbench_kv": ("__cen16_chunk16k_w4096_cen16",
                   "__cen1024_chunk16k_w4096_cen1024"),
    "scbench_vt": ("__p2_cen16_vt_chunk16k_w4096_cen16",
                   "__p2_cen1024_vt_chunk16k_w4096_cen1024"),
    "scbench_repoqa": ("__p2_cen16_rq_chunk16k_w4096_cen16",
                       "__p2_cen1024_rq_chunk16k_w4096_cen1024"),
    "scbench_prefix_suffix": ("__p2_cen16_ps_chunk16k_w4096_cen16",
                              "__p2_cen1024_ps_chunk16k_w4096_cen1024"),
}
ORDER = ["scbench_kv", "scbench_repoqa", "scbench_prefix_suffix", "scbench_vt",
         "squad", "gsm", "scbench_choice_eng", "scbench_qa_eng",
         "scbench_many_shot", "scbench_summary", "scbench_mf"]
_MUTE = contextlib.redirect_stdout(io.StringIO())


def per_sample(data, suffix, task="qa"):
    """→ ({i: 该 ratio 的均分}, {i: full__ 的均分})，照抄 parse.py 的读法。"""
    with _MUTE:
        ANSW, SUBT = parse_answer(data)
    pr, fu, i = {}, {}, 0
    while True:
        f = f"./results/{data}/{i}_{MODEL}_fastkvzip{suffix}/output-{LEVEL}.json"
        if not os.path.exists(f):
            break
        d = json.load(open(f))
        p, q, answers = [], [], []
        for fmt in [k for k in d if k.startswith(task)]:
            for info, text in d[fmt]:
                if abs(float(info[0]) - RATIO) < 1e-9:
                    p.append(text["pruned"]); q.append(text["full__"])
                answers.append(text["answer"])
        gold = ANSW[i] if ANSW else answers
        sub = SUBT[i] if SUBT else None
        if p:
            with _MUTE:
                pr[i] = float(np.mean(evaluate_answer(p, gold, data, task, subtask=sub)))
                fu[i] = float(np.mean(evaluate_answer(q, gold, data, task, subtask=sub)))
        i += 1
    return pr, fu


def boot(dif, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    dif = np.asarray(dif)
    s = dif[rng.integers(0, len(dif), (n, len(dif)))].mean(1)
    return dif.mean(), float(np.quantile(s, .025)), float(np.quantile(s, .975))


def main():
    print("\n" + "=" * 108)
    print(f"【免训练质心】Figure-11 全部 11 个面板　ratio {RATIO}　同批基线　"
          f"★=95%CI 不含 0　逐样本配对 bootstrap")
    print("-" * 108)
    print(f"{'论文面板':<21}{'n':>4}{'满缓存':>8}{'基线':>8}{'headroom':>10}"
          f"{'K=16 Δ':>22}{'K=1024 Δ':>22}")
    rows = []
    for data in ORDER:
        bs = BASE.get(data, BASE_DEFAULT)
        pb, fb = per_sample(data, bs)
        if not pb:
            print(f"{PANEL[data]:<21}  ← 基线缺失 ({bs})"); continue
        cells, ns = [], [set(pb)]
        for suf in CEN.get(data, (f"__cen16_{data}_chunk16k_w4096_cen16",
                                 f"__cen1024_{data}_chunk16k_w4096_cen1024")):
            pc, _ = per_sample(data, suf)
            cells.append(pc); ns.append(set(pc))
        common = sorted(set.intersection(*ns))
        if not common:
            print(f"{PANEL[data]:<21}  ← 无共同样本"); continue
        b = np.array([pb[i] for i in common]) * 100
        full = np.mean([fb[i] for i in common]) * 100
        out, hr = [], full - b.mean()
        for pc in cells:
            c = np.array([pc[i] for i in common]) * 100
            m, lo, hi = boot(c - b)
            out.append((m, lo, hi, c.mean()))
        s = (f"{PANEL[data]:<21}{len(common):>4}{full:>8.2f}{b.mean():>8.2f}"
             f"{hr:>+10.2f}")
        for m, lo, hi, _ in out:
            s += f"{m:>+8.2f}[{lo:>+6.1f},{hi:>+6.1f}]{'★' if (lo > 0 or hi < 0) else ' '}"
        print(s)
        rows.append((data, len(common), full, b.mean(), hr, out))

    print("-" * 108)
    for j, nm in ((0, "K=16  "), (1, "K=1024")):
        pos = sum(1 for r in rows if r[5][j][1] > 0)
        neg = sum(1 for r in rows if r[5][j][2] < 0)
        mean = np.mean([r[5][j][0] for r in rows])
        agree = sum(1 for r in rows if np.sign(r[5][j][0]) == np.sign(r[4]))
        print(f"{nm}：{len(rows)} 个面板中 显著正 {pos}，显著负 {neg}，"
              f"未分离 {len(rows)-pos-neg}；Δ 均值 {mean:+.2f}；"
              f"符号与 headroom 一致 {agree}/{len(rows)}")
    print("=" * 108)
    print("口径限制（引用时必须同时说）：")
    print(f"  · ratio {RATIO} 落在论文 Figure-11 的 x 轴（0.2–1.0）**之外**，"
          "论文没有可对照的点；打分流程与数据集相同，压缩点是我们自选的")
    print("  · 单个 ratio，不是曲线；MRCR（第 12 个面板）走 eval_chunk_mrcr.py，"
          "质心未接入 ⇒ 上限 11/12")
    print("  · n 小于 100 的面板是**数据集本身就只有那么多**"
          "（RepoQA 88 / MultiHop 90 / En.Summary 70 / ManyShot 54 / "
          "En.QA 20 / MultiChoice 18），不是被截断")


if __name__ == "__main__":
    main()
