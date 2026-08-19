#!/usr/bin/env python3
"""**完整性审计**：RESULTS_ABLATION.md 引用的每个 tag，是否跑满了整个数据集？

与 `scratch_verify_ablation.py` 的分工：那个重算**载重数字**，但只覆盖我手工列进去的
格；本脚本反过来，**从文档正文把所有 tag 抠出来**逐个查完整性，覆盖面由文档决定而不由
我的记忆决定。

三层检查，一层比一层严：

  ① 目录层：该 tag 在该数据集下有多少个样本目录，是否等于**该数据集的满量**。
     满量取自 `data/load.py` 实际加载条数（CLAUDE.md 2026-08-03 从 `loaded, #data:`
     读出）。多个 SCBench 子集天生不足 100 条，所以 n=18 是完整、不是被截断。
     **不要改成「取该 ds 下所有 tag 的最大 n」** —— 若某 ds 上每个 arm 都被同样截断，
     那种自适应口径会把系统性截断判成完整。

  ② **ratio 键层**：目录齐全 ≠ 每个 ratio 都有数。同一个作业可能只跑了部分 ratio，
     或某样本的某个 ratio 解析失败。撤回 35（`zip` 静默截断把 PrefSuf 整列右移）就是
     这一层的错，只查目录数根本看不见。这里逐 (tag, ratio) 数样本。

  ③ 例外层：确实存在**故意跑少**的作业（配额 dump 用 `--num 20`，它只产出配额记录、
     不产出任何被引用的分数）。但「例外」不能是脚本里一句白名单就算数 —— 每条例外
     必须给出**文档里写着这条说明的那句话**，脚本反查那句话是否还在。
     若有人把说明删了、却留着数字，审计立刻变红。

判据：任何被引用、n 不足、且不在**有效**例外表里的 (tag, ds[, ratio]) ⇒ FAIL。
运行：.venv/bin/python scratch_audit_complete.py
"""
import os, re, sys, glob, json

ROOT = os.path.dirname(os.path.abspath(__file__))
PRE  = os.path.join(ROOT, "external/FastKVzip/prefill")
LOGD = os.path.join(ROOT, "scratch_ctrl_logs")
# **两份文档都要扫**：`REVIEW_BRIEF.md` 是唯一对外材料，如果只扫内部记录，
# 简报里独有的 tag 就没人查。例外表的「说明句仍在」也对两份都成立即可。
DOCS = [os.path.join(ROOT, "RESULTS_ABLATION.md"),
        os.path.join(ROOT, "REVIEW_BRIEF.md")]

FULL_N = {
    "scbench_kv": 100, "scbench_prefix_suffix": 100, "scbench_mf": 100,
    "gsm": 100, "squad": 100, "scbench_vt": 90, "scbench_repoqa": 88,
    "scbench_summary": 70, "scbench_many_shot": 54, "scbench_qa_eng": 20,
    "scbench_choice_eng": 18,
}

# 声明式例外：tag → (原因, 文档里必须仍然存在的原话片段)。
# **加一条例外前先问：文档里真的没有从它读出的分数吗？** 配额 dump 作业只被用于
# 饿死率/搬动量这类协变量，那些量逐 chunk 统计、与样本数无关。
EXEMPT = {
    "_qdkv": ("配额 dump 专用，--num 20，不产出被引用的分数",
              "配额 dump 作业（`_qdvt`/`_qdps`/`_qdkv`）用 `--num 20`"),
    "_qdps": ("同上", "配额 dump 作业（`_qdvt`/`_qdps`/`_qdkv`）用 `--num 20`"),
    "_qdvt": ("同上", "配额 dump 作业（`_qdvt`/`_qdps`/`_qdkv`）用 `--num 20`"),
    "_psflr32": ("已知不完整，文档已明写并已被 `_psflr32b` 取代",
                 "`_psflr32` 只有 37 个结果目录、无 `Finished.`"),
    # `_kvf02e` 曾在此（跑到一半时被引用为「在跑」）。**跑满后主动删掉例外**——
    # 留着一条永远为真的例外，等于给那个 tag 开了一个再也不会响的警报。
}


def scan(ds):
    """→ {tag: {ratio: set(sample_idx)}}，同时返回 {tag: set(sample_idx)}（目录层）。"""
    per_ratio, per_dir = {}, {}
    for d in sorted(glob.glob(os.path.join(PRE, "results", ds, "*_chunk16k*"))):
        b = os.path.basename(d)
        m = re.match(r"^(\d+)_qwen2\.5-7b-instruct-1m_(.*)_chunk16k", b)
        if not m:
            continue
        i, rest = int(m.group(1)), m.group(2)
        for g in ("fastkvzip", "expect", "snap", "head"):
            if rest.startswith(g):
                rest = rest[len(g):]
                break
        tag = "_" + rest.lstrip("_")
        js = glob.glob(os.path.join(d, "output-*.json"))
        if not js:
            continue
        per_dir.setdefault(tag, set()).add(i)
        for f in js:
            try:
                dd = json.load(open(f))
            except Exception:
                continue
            for k in [x for x in dd if x.startswith("qa")]:
                for info, _rec in dd[k]:
                    r = round(float(info[0]), 4)
                    per_ratio.setdefault(tag, {}).setdefault(r, set()).add(i)
    return per_dir, per_ratio


def main():
    doc = "\n".join(open(f).read() for f in DOCS)
    cited = set(re.findall(r"`(_[A-Za-z0-9_.\-]+)`", doc))

    fail = 0
    dirmap, ratmap = {}, {}
    for ds in FULL_N:
        dirmap[ds], ratmap[ds] = scan(ds)

    print("== 0. 例外表是否仍被文档支持 ==")
    valid_exempt = set()
    for tag, (why, quote) in sorted(EXEMPT.items()):
        ok = quote in doc
        fail += (not ok)
        if ok:
            valid_exempt.add(tag)
        print(f"   {tag:<12} {'OK  ' if ok else '**FAIL**'} {why}")
        if not ok:
            print(f"                说明句已从文档消失：{quote!r}")

    print("\n== 1. 覆盖情况 ==")
    real = {t for t in cited if any(t in dirmap[ds] for ds in FULL_N)}
    print(f"   两份文档里的 `_xxx` 记号 {len(cited)} 个，其中 {len(real)} 个对应真实结果目录")

    print("\n== 2. 目录层：被引用且样本目录数不足满量 ==")
    n2 = 0
    for tag in sorted(real):
        for ds in FULL_N:
            if tag not in dirmap[ds]:
                continue
            n, full = len(dirmap[ds][tag]), FULL_N[ds]
            if n == full:
                continue
            ex = tag in valid_exempt
            n2 += (not ex)
            lg = os.path.join(LOGD, tag.lstrip("_") + ".log")
            fin = os.path.exists(lg) and b"Finished." in open(lg, "rb").read()
            mark = "（已声明例外）" if ex else "**违规**"
            print(f"   {tag:<18} {ds:<24} n={n:<4}/{full}  Finished={fin}  {mark}")
    fail += n2
    if n2 == 0:
        print("   **无违规**（列出的都在例外表内）" if any(True for _ in [0]) else "")

    print("\n== 3. ratio 键层：被引用的 tag，其每个 ratio 的样本数 ==")
    n3 = 0
    for tag in sorted(real):
        if tag in valid_exempt:
            continue
        for ds in FULL_N:
            rr = ratmap[ds].get(tag)
            if not rr:
                continue
            full = FULL_N[ds]
            short = {r: len(s) for r, s in sorted(rr.items()) if len(s) != full}
            if short:
                n3 += 1
                print(f"   **{tag:<18} {ds:<24} 缺 ratio:** " +
                      ", ".join(f"ρ={r} n={n}/{full}" for r, n in short.items()))
    fail += n3
    if n3 == 0:
        print("   全部 (tag, ratio) 组合都是满量")

    print("\n== 4. 基线自身（配对的公共分母） ==")
    for ds in sorted(FULL_N):
        s = dirmap[ds].get("__g8base") or dirmap[ds].get("_g8base")
        if s is None:
            continue
        n, full = len(s), FULL_N[ds]
        ok = n == full
        fail += (not ok)
        rr = ratmap[ds].get("__g8base") or ratmap[ds].get("_g8base") or {}
        badr = [r for r, v in rr.items() if len(v) != full]
        if badr:
            fail += 1
        print(f"   {ds:<24} n={n:<4}/{full}  ratio 全满={'是' if not badr else f'否 {badr}'}"
              f"   {'OK' if ok and not badr else '**FAIL**'}")

    print(f"\n{'全部通过' if fail == 0 else f'**{fail} 项 FAIL**'}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
