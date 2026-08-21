#!/usr/bin/env python3
"""从教师日志里抽自检⑤ 的逐篇余量，按**先写死的判据**给 PASS / FAIL。

自检⑤ 的量：同一个问句下，`NLL(同前缀异后缀诱饵) − NLL(正确答案)`。
为正 = 模型把真答案排在诱饵之前 = 它确实在按后缀检索。

判据（三条全过才 PASS，缺一不可）：
  ① 至少 3 篇有读数（少于 3 篇没有判断力）；
  ② **多数篇为正**（> 半数）；
  ③ **均值 > 0.02** —— 不设 0 是因为 0 附近的余量等于没有区分度，
     doc0 那次 +0.0406 已经是「勉强」，再低就不该拿来造标签。

用法：python scratch_ck5_verdict.py <log>   —— PASS 退出码 0，FAIL 退出码 1。
"""
import re
import sys

MIN_DOCS, MIN_MEAN = 3, 0.02


def margins(path):
    """→ [每篇的 (诱饵 − 正确)]。日志行形如
    `  自检⑤ 满缓存 NLL 正确 0.6347 vs 同前缀诱饵 0.6752  差 +0.0406`"""
    pat = re.compile(r"自检⑤ 满缓存 NLL 正确 ([\d.]+) vs 同前缀诱饵 ([\d.]+)")
    out = []
    for ln in open(path, errors="ignore"):
        m = pat.search(ln)
        if m:
            out.append(float(m.group(2)) - float(m.group(1)))
    return out


def main():
    path = sys.argv[1]
    ms = margins(path)
    aborted = "满缓存下模型没在检索" in open(path, errors="ignore").read()
    print(f"自检⑤ 逐篇余量（诱饵 − 正确，为正才对）：{['%+.4f' % x for x in ms]}")
    if not ms:
        print("FAIL：一条读数都没有 —— 任务或日志有问题，不是判据的事")
        return 1
    npos = sum(1 for x in ms if x > 0)
    mean = sum(ms) / len(ms)
    print(f"  篇数 {len(ms)}（需 ≥{MIN_DOCS}）・ 为正 {npos}/{len(ms)}（需 > 半数）"
          f"・ 均值 {mean:+.4f}（需 > {MIN_MEAN}）・ 中途中止 {aborted}")
    ok = (len(ms) >= MIN_DOCS and npos * 2 > len(ms)
          and mean > MIN_MEAN and not aborted)
    print("PASS：模型在满缓存下确实按后缀检索，可以造标签"
          if ok else
          "FAIL：满缓存下的区分度不够，标签会是噪声 —— 不要造表，改格式重来")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
