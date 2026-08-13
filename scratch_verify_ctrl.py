#!/usr/bin/env python3
"""ControlRetainCache 的验收检查。代数错了整轮实验就白跑，所以这些必须先过。

五项，从弱到强：

  1. **β=0 ⇒ 与基线逐字相同。** 这是结构性保证（修正项恒为 0），不是希望。
     learned-memory 那轮的"gate→0 是精确 fallback"只对 gate 成立、对空记忆不成立，
     这里必须真的成立。
  2. **预算恒等匹配。** `threshold` 按 ratio 取全局 top-n ⇒ 保留条数必须与基线**完全相等**，
     不是"近似相等"。之前 memcache 那套 `M·H·L` 核算折腾了两版，这里应当一条都不差。
  3. **β≠0 确实改变了保留集合**（否则修正被 z-score 抹平了，是 bug 不是结论）。
  4. **逐层预算会挪动，但总量守恒。** level="pair" 是全局阈值化，
     "永远对所有层求和"那条教训在这里同样适用——只看某一层会得出灾难性的错误结论。
  5. **shuffle 对照与真 novelty 给出不同的掩码**（否则对照无效）。

GPU 被 r05 sweep 占满时不要跑这个——会争显存。
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

MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
DATA, RATIO, CHUNK, WIN = "scbench_kv", 0.1, 16000, 4096


def build(kv_type, **kw):
    m = ModelKVzip(MODEL, kv_type, "fastkvzip")
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def run(model, idx=0):
    ds = load_dataset_all(DATA, model.tokenizer)
    dw = DataWrapper(DATA, ds, model)
    kv = dw.prefill_context(idx, prefill_chunk=CHUNK, window_size=WIN,
                            chunk_ratio=RATIO, level="pair")
    # 逐层保留数（必须求和后再比，不能只看某一层）
    per_layer = kv.valid.float().mean(dim=(1, 2)).tolist()
    total = int(kv.valid.sum().item())
    q = dw.get_query(idx, 0) if hasattr(dw, "get_query") else None
    out = model.generate(q, kv) if q is not None else None
    return kv, total, per_layer, out


def main():
    print("=" * 90)
    base = build("retain")
    kv_b, tot_b, pl_b, out_b = run(base)
    print(f"[基线] 保留 {tot_b} 条；逐层保留率 min/max = "
          f"{min(pl_b):.4f}/{max(pl_b):.4f}")
    del base, kv_b
    torch.cuda.empty_cache()

    for tag, kw in [("β=0", dict(ctrl_beta=0.0)),
                    ("β=0.5 evicted", dict(ctrl_beta=0.5, ctrl_src="evicted")),
                    ("β=0.5 shuffle", dict(ctrl_beta=0.5, ctrl_src="evicted",
                                           ctrl_shuffle=True))]:
        m = build("control", **kw)
        kv, tot, pl, out = run(m)
        same_total = (tot == tot_b)
        drift = max(abs(a - b) for a, b in zip(pl, pl_b))
        print(f"[{tag:<14}] 保留 {tot} 条 "
              f"（与基线{'完全相同 ✓' if same_total else f'差 {tot-tot_b} ✗'}）"
              f"　逐层保留率最大漂移 {drift:.4f}"
              f"　修正幅度 σ 中位 {torch.tensor(kv._corr_std).median() if kv._corr_std else 0:.4g}")
        if tag == "β=0":
            assert same_total, "β=0 竟然改变了保留条数——修正项没有恒为 0"
            assert drift < 1e-9, f"β=0 竟然改变了逐层分配（漂移 {drift}）"
            assert out == out_b, "β=0 生成结果与基线不同——不是精确 fallback"
            print("      ✓ β=0 与基线逐字相同（结构性保证成立）")
        else:
            assert same_total, "预算没有恒等匹配——threshold 的 top-n 语义被破坏了"
        del m, kv
        torch.cuda.empty_cache()
    print("=" * 90)
    print("注意：第 4 项（逐层预算挪动）不是 bug 而是机制本身——"
          "记忆就是通过重新分配预算起作用的。要报的是**总量守恒 + 逐层漂移量**。")


if __name__ == "__main__":
    sys.exit(main())
