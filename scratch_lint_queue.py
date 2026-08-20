#!/usr/bin/env python3
"""派发前静态检查 `/tmp/vq/jobs.txt` —— 零 GPU。

**为什么必须有**：2026-08-20 我一次排了 5 个 ps 作业，注入表文件名
`scratch_quota_dbh_ps01.npy` 是照着 kv 的命名**编**的、根本不存在，
5 个作业各占一个 GPU 槽跑到 prune_chunk 才 FileNotFoundError 挂掉。
失败是响亮的（符合"静默回退一律改抛错"），但**本可以在派发前就发现**。

检查项（每一项都是已经真实踩过、或会静默跑成另一个实验的）：
  1. 注入表存在
  2. 表最后一维 == L*H（模型相关：Qwen2.5-7B-1M 112、Qwen3-8B 288）
  3. tag 未跑过（重复排队）
  4. mode 在已知集合内
  5. order=band ⇒ 必须有 COV_BAND 与 COV_N，且层带在 [0, L-1] 内
  6. Qwen3 ⇒ 只允许 floor/floorcov（未训练打分器）
  7. 退化：clen > window/ratio。clen 未知时**计数并报告**，不静默跳过。
"""
import os, sys, glob

WINDOW = 4096
GEOM = {"Qwen/Qwen2.5-7B-Instruct-1M": (28, 4), "Qwen/Qwen3-8B": (36, 8)}
MODES = {"full", "within", "across", "floor", "floorproj", "pathproj",
         "floorpath", "maxlift", "floorcov", "-", ""}
# 实测 clen（tokenizer 口径见 RESULTS_ABLATION）。键 = (模型简称, 数据集)
# 上下文长度表。**必须覆盖全部 11 个 panel** —— 缺一个，该行的退化检查就被
# 静默跳过（脚本会计数并印「不是通过，是没查」，但那行看起来仍是 OK）。
# 2026-08-20 补全前只有 4 个 Qwen2.5 条目，全网格自检 4 行里跳过 3 行。
# 数值来自 `data/load.py:get_data_list` 的注释，`prefix_suffix` 与 `kv`
# 另有实测核对（112,577 / 169,428）。
CLEN = {("qwen2.5", "gsm"): 86,
        ("qwen2.5", "squad"): 203,
        ("qwen2.5", "scbench_many_shot"): 26474,
        ("qwen2.5", "scbench_repoqa"): 72499,
        ("qwen2.5", "scbench_prefix_suffix"): 112577,
        ("qwen2.5", "scbench_summary"): 117806,
        ("qwen2.5", "scbench_choice_eng"): 119299,
        ("qwen2.5", "scbench_qa_eng"): 122101,
        ("qwen2.5", "scbench_vt"): 124551,
        ("qwen2.5", "scbench_mf"): 149860,
        ("qwen2.5", "scbench_kv"): 169428,
        ("qwen3", "scbench_kv"): 24452}          # Qwen3 换 _short，取实测 min
ROOT = os.path.dirname(os.path.abspath(__file__))


def main(path="/tmp/vq/jobs.txt"):
    import numpy as np
    def _finished(p):
        # 只读尾部 4 KB —— 有的日志几百 MB，整读一遍会让 lint 变慢到没人愿意跑。
        with open(p, "rb") as fh:
            fh.seek(max(0, os.path.getsize(p) - 4096))
            return b"\nFinished." in b"\n" + fh.read()
    done = {os.path.basename(p)[:-4]
            for p in glob.glob(f"{ROOT}/scratch_ctrl_logs/*.log") if _finished(p)}
    bad = unknown_clen = 0
    for ln, raw in enumerate(open(path), 1):
        f = raw.split()
        if not f:
            continue
        # ratio 字段可以是**逗号分隔的多个** —— qrun.sh 原样传给 VARIKV_RATIOS，
        # 全网格扫描就是一个 panel 一个作业带 6–7 个 ratio。旧版 `float(f[1])`
        # 会在这种行上直接抛 ValueError。
        ds, tag, tab, num = f[0], f[2], f[3], f[4]
        ratios = [float(x) for x in f[1].split(",") if x]
        ratio = max(ratios)          # 单值行与旧行为逐位相同
        mode = f[6] if len(f) > 6 else ""
        xenv = f[7] if len(f) > 7 else ""
        model = f[8] if len(f) > 8 else "Qwen/Qwen2.5-7B-Instruct-1M"
        env = dict(kv.split("=", 1) for kv in xenv.split(",") if "=" in kv)
        L, H = GEOM.get(model, (None, None))
        errs = []
        warns = []
        if L is None:
            errs.append(f"未知模型 {model}（几何表里没有，无法查表大小/层带）")
        if mode not in MODES:
            errs.append(f"未知 mode={mode}")
        if tab != "-":
            p = os.path.join(ROOT, tab)
            if not os.path.exists(p):
                errs.append(f"注入表不存在: {tab}")
            elif L is not None:
                shp = np.load(p).shape
                if shp[-1] != L * H:
                    errs.append(f"表最后一维 {shp[-1]} != L*H={L*H}（{model}）")
        if tag.lstrip("_") in done:
            errs.append(f"tag 已跑完过（重复排队）")
        if env.get("VARIKV_COV_ORDER") == "bandrand" and "VARIKV_COV_SEED" not in env:
            errs.append("order=bandrand 但缺 VARIKV_COV_SEED")
        if env.get("VARIKV_COV_ORDER") in ("band", "bandrand"):
            if "VARIKV_COV_BAND" not in env:
                errs.append("order=band 但缺 VARIKV_COV_BAND")
            elif L is not None:
                lo, hi = (int(x) for x in env["VARIKV_COV_BAND"].split("-"))
                if not (0 <= lo <= hi < L):
                    errs.append(f"层带 {lo}-{hi} 越界（L={L}）")
            if "VARIKV_COV_N" not in env:
                errs.append("order=band 但缺 VARIKV_COV_N")
        if "qwen3" in model.lower() and mode not in ("floor", "floorcov"):
            errs.append(f"Qwen3 用未训练打分器，只允许 floor/floorcov（收到 {mode}）")
        key = ("qwen3" if "qwen3" in model.lower() else "qwen2.5", ds)
        if key in CLEN:
            # 逐 ratio 判退化。**部分退化是预期的**（全网格里 gsm/squad 在所有 ρ 上
            # 都退化，ManyShot 在 ρ≤0.1 退化），表里标 `°`/`~`，不该整行判错；
            # 只有**全部 ratio 都退化**时这个作业才是纯粹白跑。
            # ⚠ 2026-08-20 实测更正：**退化不等于控制器臂是 no-op**。
            # CLAUDE.md 记的「ratio·clen < window ⇒ 保留集恒等于局部窗口 ⇒
            # 任何分数扰动都是构造性 no-op」是就**基线驱逐路径**说的。
            # 但对控制器臂，逐样本实测：
            #     squad@ρ=0.1  scalar 臂与基线 **52/100 条分数不同**
            #     squad@ρ=0.2  21/100；gsm@ρ=0.1  17/100
            # 且现有全网格里 squad@0.1 的 scalar 是 **+3.44±0.45★(2)**。
            # 机制未查明（controller 走 learned_ctrlcache 自己的 argsort/配额路径，
            # 可能绕开 wrapper 的退化分支）。**在查明之前一律只警告、不拦截** ——
            # 拦下来会静默丢掉已被证明非零的格子。
            deg = [r for r in ratios if CLEN[key] <= WINDOW / r]
            if deg:
                scope = "全部 ρ" if len(deg) == len(ratios) else "部分 ρ"
                warns.append(f"{scope}满足 clen <= window/ρ（表里应标 °/~）: "
                             f"{','.join(str(r) for r in deg)}"
                             f"；**但控制器臂实测非 no-op，不拦截**")
        else:
            unknown_clen += 1
        bad += bool(errs)
        mark = "**FAIL**" if errs else ("WARN" if warns else "OK")
        rs = f[1] if len(f[1]) <= 22 else f[1][:19] + "..."
        print(f"  L{ln:<3} {tag:<14} {ds:<24} ρ={rs:<23} {mark}")
        for e in errs:
            print(f"        ! {e}")
        for w in warns:
            print(f"        ~ {w}")
    print(f"\nclen 未知、退化检查跳过的行数: {unknown_clen}"
          f"（**不是通过，是没查**）")
    print("全部通过" if bad == 0 else f"**{bad} 行 FAIL**")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
