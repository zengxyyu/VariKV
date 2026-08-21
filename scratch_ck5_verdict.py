#!/usr/bin/env python3
"""从教师日志里抽自检⑤ 的逐篇余量，按**先写死的判据**给 PASS / FAIL。

自检⑤A 的量：**满缓存下**同一个问句的 `NLL(诱饵) − NLL(正确答案)`。
**只认 A 行**：B 行是 ρ 压缩后的读数，模型在那里分不开**正是 headroom**，
不是拒绝理由 —— 早先把两者混为一谈，错杀过一版格式。
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
    # **只认 A 行（满缓存）** —— B 行是 ρ 下的读数，那是 headroom 不是闸。
    # 旧格式 `自检⑤ 满缓存 NLL 正确 X vs 同前缀诱饵 Y` 也认，向后兼容旧日志。
    pat = re.compile(r"自检⑤A 满缓存 NLL 正确 ([\d.]+) vs 诱饵 ([\d.]+)"
                     r"|自检⑤ 满缓存 NLL 正确 ([\d.]+) vs 同前缀诱饵 ([\d.]+)")
    out = []
    for ln in open(path, errors="ignore"):
        m = pat.search(ln)
        if m:
            g = m.groups()
            a_, b_ = (g[0], g[1]) if g[0] else (g[2], g[3])
            out.append(float(b_) - float(a_))
    return out


def pick(paths):
    """多份日志里挑赢家：**先要 PASS，再比均值**。全不 PASS 就返回 None。

    为什么不是「直接比均值取最大」：均值最大的那个完全可能仍是负的。
    判据的顺序是「先合格、再择优」，不能倒过来。
    """
    best = None
    for q in paths:
        ms = margins(q)
        if not ms:
            print(f"  {q}: 无读数"); continue
        aborted = "满缓存下模型没在检索" in open(q, errors="ignore").read()
        npos, mean = sum(1 for x in ms if x > 0), sum(ms) / len(ms)
        ok = (len(ms) >= MIN_DOCS and npos * 2 > len(ms)
              and mean > MIN_MEAN and not aborted)
        print(f"  {q}: 篇 {len(ms)} 为正 {npos} 均值 {mean:+.4f} "
              f"{'PASS' if ok else 'fail'}")
        if ok and (best is None or mean > best[1]):
            best = (q, mean)
    return best


def main():
    if sys.argv[1] == "--pick":
        print("格式挑选（先合格、再择优）：")
        b = pick(sys.argv[2:])
        if b is None:
            print("PICK=NONE"); print("全部不合格 —— 不要造表，继续改格式"); return 1
        print(f"PICK={b[0]}"); print(f"赢家均值 {b[1]:+.4f}"); return 0
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
