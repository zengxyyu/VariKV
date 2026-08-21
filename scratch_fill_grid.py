#!/usr/bin/env python3
"""把静态 `u` 表的全网格（11 panel × 6 ratio）按**任务族**排进队列。

为什么按族排：教师的合成任务是 key→value 检索，与 Retrieval 族同构、与
Redundancy 族最远。先跑同族的，能最快判出「收益是否只在同族内」——这正是
`Δ = a·headroom + b` 里那三个差一个数量级的斜率提出的问题。

三条硬规则：
  · **已有结果的格不重排**（按结果目录判，不按日志）；
  · **队列里已有的不重排**（防止同一格并行跑两遍，已犯过一次）；
  · **结构性退化的格不排** —— `ratio·clen ≤ window` 时保留集恒等于局部窗口、
    与分数无关，任何配额注入都是构造性 no-op（CLAUDE.md 的 silent-degeneracy
    陷阱）。这里用 `scratch_lint_queue` 同一份 runtime 实测判定，不套公式。
"""
import os
import subprocess
import sys

ROOT = "/home/ubuntu/zxy/vlm-memory"
RES = os.path.join(ROOT, "external/FastKVzip/prefill/results")
Q = "/tmp/vq/jobs.txt"
MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
CKPT = "../../../varikv/chead10_s0.pt/memoryless.pt"

# **全部从 `scratch_grid_spec` 派生** —— 此前这里与 `scratch_utab_report.py`
# 各抄一份清单，排出去的新格跑完不进表（KV@0.4/@0.75、GSM8K@0.1 躺了一轮）。
from scratch_grid_spec import PANELS as _P, RATIOS, NUM, FAM, tag as _mktag  # noqa: E402

PANELS = [(d, c, NUM[d], FAM[d]) for d, c, _, _, _ in _P]
# 成本序（同族内先跑便宜的）：样本数 x 上下文 ktoken 的粗略乘积
_CTX = {"scbench_repoqa": 72, "scbench_kv": 169, "scbench_prefix_suffix": 113,
        "scbench_choice_eng": 119, "scbench_qa_eng": 122, "gsm": 1, "squad": 1,
        "scbench_mf": 150, "scbench_summary": 118, "scbench_many_shot": 26,
        "scbench_vt": 125}
COST = {d: NUM[d] * _CTX[d] for d, _, _, _, _ in _P}


def have_result(ds, tag):
    """有没有跑过 —— 看结果目录，不看日志（日志可能是别的 tag 的残留）。"""
    import glob
    return bool(glob.glob(os.path.join(RES, ds, f"*_{tag}_*")))


def degenerate(ds, r):
    """沿用 lint 的 runtime 实测判定；判不了就返回 None（当作可跑，但会被 lint 拦）。"""
    try:
        sys.path.insert(0, ROOT)
        from scratch_all_report import _degenerate
        return _degenerate(ds, r)
    except Exception:
        return None


def main():
    table = sys.argv[1] if len(sys.argv) > 1 else "varikv/tab_u_l2.npy"
    pre = sys.argv[2] if len(sys.argv) > 2 else "_uq01"
    dry = "--dry" in sys.argv
    existing = ""
    if os.path.exists(Q):
        existing = open(Q).read()
    rows, skip_done, skip_q, skip_deg = [], [], [], []
    for fam in ("retrieval", "qa", "redund"):
        fam_rows = [p for p in PANELS if p[3] == fam]
        fam_rows.sort(key=lambda p: COST[p[0]])
        for ds, code, num, _ in fam_rows:
            for r, rc in RATIOS:
                tag = _mktag(ds, r, pre)
                if have_result(ds, tag):
                    skip_done.append(tag); continue
                if f" {tag} " in existing:
                    skip_q.append(tag); continue
                if degenerate(ds, r) is True:
                    skip_deg.append(f"{tag}(ρ={r})"); continue
                rows.append(f"{ds} {r} {tag} {table} {num} - full "
                            f"VARIKV_QUOTA_RELMB=0.01 {MODEL} {CKPT}")
    print(f"已有结果 {len(skip_done)} 格 ・ 已在队列 {len(skip_q)} 格 ・ "
          f"结构性退化跳过 {len(skip_deg)} 格 ・ **新排 {len(rows)} 格**")
    if skip_deg:
        print("  退化跳过：" + " ".join(skip_deg))
    for x in rows[:6]:
        print("  +", " ".join(x.split()[:3]))
    if len(rows) > 6:
        print(f"  … 共 {len(rows)} 条")
    if dry:
        print("（--dry，未写入）"); return
    with open(Q, "a") as f:
        for x in rows:
            f.write(x + "\n")
    print(f"已写入 {Q}")


if __name__ == "__main__":
    main()
