#!/usr/bin/env python3
"""ControlRetainCache 的验收检查。代数错了整轮实验就白跑，所以这些必须先过。

**上一版这个脚本本身有 bug，五项里真正被检验的只有两项。**
- `hasattr(dw, "get_query")` 恒为 False —— `get_query` 是 `data/wrapper.py:9` 的**模块级
  函数**，不是 `DataWrapper` 的方法（它的方法叫 `generate_answer`）。于是 `out` 和 `out_b`
  都是 `None`，`assert out == out_b` 退化成 `assert None == None`，恒真。
  「β=0 生成逐字相同」这条**从未真正执行过**。
- 只比了逐层保留**数量**。两个掩码完全可以每层留一样多、留的却是完全不同的 token
  （`100101` vs `011100`）。数量相等**远弱于**掩码相等。

现在改成直接对 `valid` 做逐位比较。五项：

  1. β=0 ⇒ 掩码与基线**逐位相同**，且生成结果逐字相同。
  2. 预算相等 —— 但这是**经验事实不是构造性保证**：父类用 `score > score_sort[n]` 而不是
     严格 topk，阈值处并列会让保留数少于 n。所以这里是"实测是否相等"。
  3. β≠0 ⇒ 掩码确实改变（否则修正被 z-score 抹平了，是 bug 不是结论）。
  4. 逐层预算会挪动但总量守恒 —— 这不是 bug 而是机制本身；报总量 + 逐层漂移。
     "永远对所有层求和"那条教训在这里同样适用。
  5. shuffle 对照与真 novelty 给出**不同**的掩码（否则对照无效）。

GPU 被别的 sweep 占满时不要跑——会争显存。
"""
import os
import sys

import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
os.chdir(os.path.join(ROOT, "external/FastKVzip/prefill"))

from data.load import load_dataset_all                      # noqa: E402
from data.wrapper import DataWrapper                        # noqa: E402
from model.wrapper import ModelKVzip                        # noqa: E402
from utils import Evaluator, set_gen_length                 # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
DATA, RATIO, CHUNK, WIN, IDX = "scbench_kv", 0.1, 16000, 4096, 0


def run(kv_type, **kw):
    """完全照 eval_chunk.py:129-146 的真实路径取生成结果。

    `generate_answer` 返回的是 `(inputs, info)` **元组**而不是文本——真正的生成要
    再经 `Evaluator(model, inputs, info)(kv, generate=True)`，返回 {fmt: {pruned,
    full__, answer}}。直接拿 `generate_answer` 的返回值去比较是比错了对象。
    """
    m = ModelKVzip(MODEL, kv_type, "fastkvzip")
    for k, v in kw.items():
        setattr(m, k, v)
    ds = load_dataset_all(DATA, m.tokenizer)
    dw = DataWrapper(DATA, ds, m)
    set_gen_length(DATA, m)
    kv_full = dw.prefill_context(IDX, do_score=False)
    inputs, info = dw.generate_answer(IDX, kv_full, prob=False)
    ev = Evaluator(m, inputs, info)
    del kv_full
    torch.cuda.empty_cache()

    kv = dw.prefill_context(IDX, prefill_chunk=CHUNK, window_size=WIN,
                            chunk_ratio=RATIO, level="pair")
    valid = kv.valid.clone()                                  # [L,H,n] bool
    res = ev(kv, generate=True)
    gen = {f: v["pruned"] for f, v in res.items()}            # 只比压缩臂的生成
    diag = dict(corr=list(getattr(kv, "corr_std", [])),
                cv=list(getattr(kv, "norm_cv", [])))
    del m, kv, dw, ds, ev
    torch.cuda.empty_cache()
    return valid, gen, diag


def summarize(v):
    return int(v.sum()), v.float().mean(dim=(1, 2))           # 总数、逐层保留率


def main():
    print("=" * 94)
    vb, ob, _ = run("retain")
    tot_b, pl_b = summarize(vb)
    print(f"[基线]        保留 {tot_b} 条　逐层保留率 {pl_b.min():.4f}–{pl_b.max():.4f}")
    print(f"              生成 {len(ob)} 个 format，首个前 60 字: {str(list(ob.values())[0])[:60]!r}")

    results = {}
    for tag, kw in [("β=0", dict(ctrl_beta=0.0)),
                    ("β=+0.5 evicted", dict(ctrl_beta=0.5, ctrl_src="evicted")),
                    ("β=+0.5 shuffle", dict(ctrl_beta=0.5, ctrl_src="evicted",
                                            ctrl_shuffle=True)),
                    ("β=+0.5 retained", dict(ctrl_beta=0.5, ctrl_src="retained"))]:
        v, o, dg = run("control", **kw)
        tot, pl = summarize(v)
        diff = int((v ^ vb).sum())                            # 逐位不同的条目数
        drift = float((pl - pl_b).abs().max())
        print(f"[{tag:<15}] 保留 {tot} 条（基线 {tot_b}，差 {tot-tot_b:+d}）"
              f"　掩码逐位不同 {diff}　逐层漂移 {drift:.4f}"
              f"　修正σ {torch.tensor(dg['corr']).median() if dg['corr'] else 0:.4g}")
        results[tag] = (v, o, tot, diff)

    print("-" * 94)
    v0, o0, tot0, diff0 = results["β=0"]
    assert diff0 == 0, f"❌ β=0 的掩码与基线有 {diff0} 位不同——修正项没有恒为 0"
    assert o0 == ob, "❌ β=0 生成结果与基线不同——不是精确 fallback"
    print("✓ 1. β=0 掩码逐位相同 且 生成逐字相同")
    print(f"{'✓' if tot0 == tot_b else '⚠'} 2. 预算：β=0 {tot0} vs 基线 {tot_b}"
          + ("（相等）" if tot0 == tot_b else "（不等——阈值处有并列，说明"
             "'构造性恒等'的说法确实过强）"))

    ve, _, tote, diffe = results["β=+0.5 evicted"]
    assert diffe > 0, "❌ β≠0 竟然没有改变掩码——修正被 z-score 抹平了"
    print(f"✓ 3. β≠0 改变了掩码（{diffe} 位，占 {diffe/vb.numel():.3%}）")
    print(f"{'✓' if tote == tot_b else '⚠'} 4. 总量：{tote} vs {tot_b}"
          f"　（逐层会挪动是**机制本身**，要看的是总量守恒）")

    vs, _, _, _ = results["β=+0.5 shuffle"]
    dshuf = int((vs ^ ve).sum())
    assert dshuf > 0, "❌ shuffle 与真 novelty 掩码相同——随机对照无效"
    print(f"✓ 5. shuffle 与真 novelty 掩码不同（{dshuf} 位）")

    vr, _, _, _ = results["β=+0.5 retained"]
    print(f"   补充：retained 源与 evicted 源掩码相差 {int((vr ^ ve).sum())} 位"
          f"（两者语义相反，本就该不同）")
    print("=" * 94)


if __name__ == "__main__":
    sys.exit(main())
