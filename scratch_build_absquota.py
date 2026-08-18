#!/usr/bin/env python3
"""把 `VARIKV_QUOTA_DUMP` 的 jsonl 转成逐样本逐 chunk 的绝对配额 npz。

用于**跨方法精确移植**：同一样本先跑捐赠方存下 `b^donor_{sample,chunk,l,h}`，
再用接收方的排序按完全相同的配额重放 ⇒ 配额逐位相同，唯一变量是排序。

为什么不用跨文档平均表：对同一 chunk 位置，不同文档的捐赠方配额本就不同，
平均后再喂给文档 1，结果差**无法归因** —— 分不清「接收方排序不好」还是
「没给它这个文档真正的配额」。

npz 字段（`lo`/`hi` 是对齐校验用的，注入端每个 chunk 都断言匹配）：
    quota  [S, C, 112] int32     绝对配额
    lo,hi  [S, C]      int64     该 chunk 的 evict_range，注入端逐 chunk 校验
    nchunk [S]         int32     每个样本的实际 chunk 数
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.src)]
    rows = [r for r in rows if isinstance(r["b_base"], list) and len(r["b_base"]) == 112]
    # 样本边界：dump 的 `seq` 是**样本内**的 chunk 序号（cache 每样本重建后从 1 起）
    samples, cur = [], []
    for r in rows:
        if r["seq"] == 1 and cur:
            samples.append(cur); cur = []
        cur.append(r)
    if cur:
        samples.append(cur)

    S = len(samples)
    C = max(len(x) for x in samples)
    quota = np.zeros((S, C, 112), np.int32)
    lo = np.full((S, C), -1, np.int64)
    hi = np.full((S, C), -1, np.int64)
    nch = np.zeros(S, np.int32)
    for i, smp in enumerate(samples):
        nch[i] = len(smp)
        for j, r in enumerate(smp):
            quota[i, j] = np.asarray(r["b_base"], np.int32)
            lo[i, j], hi[i, j] = int(r["lo"]), int(r["hi"])
    np.savez(a.out, quota=quota, lo=lo, hi=hi, nchunk=nch)
    print(f"{a.src} -> {a.out}")
    print(f"  {S} 样本 × 最多 {C} chunk；每样本 chunk 数 "
          f"{sorted(set(nch.tolist()))}")
    print(f"  Σquota 逐样本均值 {quota.sum(2).sum(1).mean():.0f}"
          f"（逐 chunk 均值 {quota.sum(2)[quota.sum(2) > 0].mean():.0f}）")
    print(f"  零配额头占比 {(quota[quota.sum(2) > 0] == 0).mean() * 100:.1f}%")


if __name__ == "__main__":
    raise SystemExit(main())
