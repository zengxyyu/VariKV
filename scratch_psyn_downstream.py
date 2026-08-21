#!/usr/bin/env python3
"""合成 PrefSuf 教师落地后的下游一条龙：**建表 → 排评测**，带三道硬闸。

为什么写成脚本而不是手敲：手敲会跳过检查。这里三道闸全部**先写死**：

  闸① 教师产物必须完整      —— 记录数 ≥ 期望的九成，且篇数 ≥ 8；
  闸② 靶子必须可辨识        —— 直接反解 u 的 5 折 CV **R² ≥ 0.05** 且 **λ 未顶到上限**。
      理由：chunk 标签在 30 篇 14,400 条上 R² 只有 +0.004、λ 恒为 100，
      造出来的表是噪声；而 bulk 的 kv 教师是 +0.3966、λ=10（§十之十）。
  闸③ 新表必须与旧表不同    —— 若两张表的 Spearman > 0.95，说明换任务没换出
      新东西，评测下去只是把旧结论再花一遍 GPU。

任一闸不过 ⇒ 不建表、不排队、打印原因退出。

排队顺序按用户口径：**先评旧表表现差的相似任务**。
  ① Retr.PrefSuf 全 6 个 ratio —— 靶子本身（旧表 −0.40/−6.40★/−14.20★/−14.00★/−10.60★/+1.80）
  ② Retr.KV @0.1/@0.2 —— 旧表最强的两格，看新表**有没有把赢的地方弄丢**
  ③ En.QA@0.2（旧表 −6.42★）、Retr.MultiHop@0.3（旧表 −2.62★）—— 另两处显著为负
  ④ 其余 panel 全 ratio
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = "/home/ubuntu/zxy/vlm-memory"
Q = "/tmp/vq/jobs.txt"
M = "Qwen/Qwen2.5-7B-Instruct-1M"
C = "../../../varikv/chead10_s0.pt/memoryless.pt"
OLD = "varikv/tab_u_l2.npy"
sys.path.insert(0, ROOT)

# (dataset, 短码, 满量)
PAN = {"scbench_prefix_suffix": ("psr", 100), "scbench_kv": ("r", 100),
       "scbench_qa_eng": ("qa", 20), "scbench_vt": ("vt", 90),
       "scbench_repoqa": ("rq", 88), "scbench_choice_eng": ("ce", 18),
       "gsm": ("gsm", 100), "squad": ("sq", 100), "scbench_mf": ("mf", 100),
       "scbench_summary": ("sm", 70), "scbench_many_shot": ("ms", 54)}
RAT = [(0.1, "01"), (0.2, "02"), (0.3, "03"), (0.4, "04"), (0.5, "05"), (0.75, "075")]
# 优先级：先旧表输得最惨的靶子，再检查旧表赢的地方有没有丢
PRIO = ([("scbench_prefix_suffix", r) for r, _ in RAT]
        + [("scbench_kv", 0.1), ("scbench_kv", 0.2)]
        + [("scbench_qa_eng", 0.2), ("scbench_vt", 0.3)])


def gate1(recs, n_doc=10, n_dir=256):
    want = n_doc * n_dir
    docs = len({x["doc"] for x in recs})
    ok = len(recs) >= 0.9 * want and docs >= max(2, int(0.8 * n_doc))
    print(f"闸① 完整性：{len(recs)} 条 / {docs} 篇"
          f"（期望 ~{want} 条、≥{max(2, int(0.8 * n_doc))} 篇）"
          f" ⇒ {'过' if ok else '**不过**'}")
    return ok


LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]


def _fit(recs, lams=LAMS):
    """岭回归反解 u，λ 由 5 折 CV 选。**网格延到 1e5** —— 原来只到 100，
    psyn 的最优 λ 是 1000，于是「λ 顶到上限」被误当成「没信号」。"""
    X = np.array([r["d"] for r in recs], float)
    y = np.array([r["A"] for r in recs], float)
    X = X - X.mean(0)
    sx = X.std() or 1.0
    Xn = X / sx
    best = (-9.0, None, None)
    for l_ in lams:
        rs = np.random.default_rng(0)
        idx = rs.permutation(len(y))
        pred = np.zeros_like(y)
        for f in range(5):
            te = idx[f::5]
            tr = np.setdiff1d(idx, te)
            w = np.linalg.solve(Xn[tr].T @ Xn[tr] + l_ * np.eye(Xn.shape[1]),
                                Xn[tr].T @ (y[tr] - y[tr].mean()))
            pred[te] = Xn[te] @ w + y[tr].mean()
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        if r2 > best[0]:
            u = np.linalg.solve(Xn.T @ Xn + l_ * np.eye(Xn.shape[1]),
                                Xn.T @ (y - y.mean())) / sx
            best = (r2, l_, u - u.mean())
    return best


def gate2(recs):
    """**2026-08-21 重写：原判据错了两处。**

    错一：拿 `λ < 100` 当「有信号」的代理。λ=100 只是原网格的**上端**；
          延长到 1e5 后 psyn 的最优 λ 是 **1000、是内点极值**，R²=+0.0939。
          正确的写法是「最优 λ 必须是**内点**」——端点意味着网格没盖住。
    错二：把 `‖u‖₂` 小当问题。`build_delta` 会把 u 归一化成
          `½‖Δb‖₁ = mb·B`，**只有方向进表**，范数不可辨识。

    另加一条**不当闸的登记量**：逐篇 u 的两两 Spearman。静态表的前提是方向
    跨文档可迁移，这个量直接测它。kv 教师是 **+0.1630（6/6 为正）**，
    是已知能拿 +29.40 的水平；低于它很多就该**先登记失败预测再去实测**。
    """
    from scipy.stats import spearmanr
    r2, lam, u = _fit(recs)
    interior = lam not in (LAMS[0], LAMS[-1])
    ok = r2 >= 0.05 and interior
    print(f"闸② 可辨识：CV R²={r2:+.4f}（需 ≥0.05）  最优 λ={lam:g}"
          f"（需内点，实为{'内点' if interior else '**端点**'}）"
          f"  参照 kv 教师 R²=+0.3966/λ=10 ⇒ {'过' if ok else '**不过**'}")
    docs = sorted({x["doc"] for x in recs})
    us = [_fit([x for x in recs if x["doc"] == d])[2] for d in docs
          if len([x for x in recs if x["doc"] == d]) >= 100]
    if len(us) >= 2:
        ss = [spearmanr(us[i], us[j]).statistic
              for i in range(len(us)) for j in range(i + 1, len(us))]
        npos = sum(1 for x in ss if x > 0)
        print(f"  ⚠ 登记量（**不当闸**）逐篇 u 两两 Spearman："
              f"{len(us)} 篇 {len(ss)} 对，均值 {np.mean(ss):+.4f}，"
              f"为正 {npos}/{len(ss)}　参照 kv 教师 +0.1630（6/6）")
        if np.mean(ss) < 0.10:
            print("  ⇒ **先登记预测：方向跨文档迁移性远低于 kv 教师，"
                  "静态表大概率不成立。** 仍然建表评测，由实测裁决 —— "
                  "kv 表当初也只有 +0.163 却拿到 +29.40，所以不能只凭这个量判死。")
    return ok, u, r2


def gate3(new_npy):
    from scipy.stats import spearmanr
    old = np.load(os.path.join(ROOT, OLD)).ravel()
    new = np.load(new_npy).ravel()
    if old.shape != new.shape:
        print(f"闸③ 形状不同 {old.shape} vs {new.shape} —— 当作不同，过")
        return True
    s = spearmanr(old, new).statistic
    ok = abs(s) <= 0.95
    print(f"闸③ 与旧表的 Spearman = {s:+.4f}（需 |·| ≤0.95，否则没换出新东西）"
          f" ⇒ {'过' if ok else '**不过**'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="scratch_adv_grad_psyn.json")
    ap.add_argument("--out", default="varikv/tab_u_psyn_l2.npy")
    ap.add_argument("--pre", default="_up01")
    ap.add_argument("--dry", action="store_true")
    # 期望规模做成参数，是为了能用**已知可辨识**的 kv 教师产物自测闸②③ ——
    # 一个从没在正样本上跑过的判据，不知道它会不会永远说「不过」。
    ap.add_argument("--expect_doc", type=int, default=10)
    ap.add_argument("--expect_dir", type=int, default=256)
    a = ap.parse_args()

    jp = os.path.join(ROOT, a.json)
    if not os.path.exists(jp):
        print(f"教师产物还不存在：{a.json} —— 等它跑完再来"); return 2
    recs = json.load(open(jp))
    if not gate1(recs, a.expect_doc, a.expect_dir):
        return 1
    ok2, _, _ = gate2(recs)
    if not ok2:
        print("**靶子不可辨识 ⇒ 不建表。** 与 chunk 标签同一个失败形态，"
              "加样本量无效，要改任务格式。")
        return 1

    outp = os.path.join(ROOT, a.out)
    cmd = [os.path.join(ROOT, ".venv/bin/python"),
           os.path.join(ROOT, "scratch_u_to_table.py"),
           "--json", jp, "--alloc", "l2", "--out", outp]
    print("建表：" + " ".join(cmd[1:]))
    if a.dry:
        print("（--dry，到此为止）"); return 0
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1200:] or r.stderr[-1200:])
    if r.returncode != 0 or not os.path.exists(outp):
        print("**建表失败**"); return 1
    if not gate3(outp):
        return 1

    import glob
    RES = os.path.join(ROOT, "external/FastKVzip/prefill/results")
    ex = open(Q).read() if os.path.exists(Q) else ""
    rows, done = [], 0
    rest = [(d, r) for d in PAN for r, _ in RAT if (d, r) not in PRIO]
    for ds, r in PRIO + rest:
        code, num = PAN[ds]
        rc = dict(RAT)[r]
        tag = f"{a.pre}{code}{rc}"
        if glob.glob(os.path.join(RES, ds, f"*_{tag}_*")) or f" {tag} " in ex:
            done += 1; continue
        rows.append(f"{ds} {r} {tag} {os.path.relpath(outp, ROOT)} {num} - full "
                    f"VARIKV_QUOTA_RELMB=0.01 {M} {C}")
    # **插到队首**：用户要求新表的相似任务优先于 kv 表的全网格。
    # **必须拿 poplock** —— worker 的 pop() 是「读首行 + sed 删除」两步，
    # 我这里若同时整体重写，会把它正要删的那行又写回去、或吞掉它刚写的状态。
    # worker.sh 用的就是 mkdir /tmp/vq/poplock 这把锁。
    import time
    PL = "/tmp/vq/poplock"
    got = False
    for _ in range(200):
        try:
            os.mkdir(PL); got = True; break
        except FileExistsError:
            time.sleep(0.3)
    if not got:
        print("**拿不到 poplock，放弃写队列**（避免与 worker 竞争丢行）"); return 1
    try:
        old_q = open(Q).read() if os.path.exists(Q) else ""
        with open(Q, "w") as f:
            f.write("\n".join(rows) + ("\n" if rows else "") + old_q)
    finally:
        os.rmdir(PL)
    print(f"排入队首 {len(rows)} 格（跳过已有 {done}）：前 8 个 = "
          + " ".join(x.split()[2] for x in rows[:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
