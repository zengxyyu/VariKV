"""独立复核 RESULTS_GRID.md：用 scratch_read_scores.py 重算，与表里的数字逐格比。

**为什么必须跨实现复核**：表由 `scratch_all_report.py` 生成，它自带一份 glob 与
bootstrap；若那份实现有系统性偏差，重跑同一个脚本永远发现不了。这里用
`scratch_read_scores.py`（另一套 glob + 另一套 bootstrap 调用）独立重算。
"""
import re, sys
import numpy as np
sys.path.insert(0, '/home/ubuntu/zxy/vlm-memory')
from scratch_read_scores import read_scores, paired

PANEL = {"Retr.KV": "scbench_kv", "Retr.PrefSuf": "scbench_prefix_suffix",
         "Code.RepoQA": "scbench_repoqa", "SQuAD": "squad", "GSM8K": "gsm",
         "En.QA": "scbench_qa_eng", "En.MultiChoice": "scbench_choice_eng",
         "En.Summary": "scbench_summary", "Retr.MultiHop": "scbench_vt",
         "Math.Find": "scbench_mf", "ICL.ManyShot": "scbench_many_shot"}
TPL = {"v2": "_g8v2", "v3": "_g8v3",
       "v2c": "_v2c_s{S}", "scalar": "_d10scalar_s{S}", "kv": "_d10kv_s{S}"}
SEEDED = {"v2c", "scalar", "kv"}
RAT = [0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]

# 解析表格
rows, panel = [], None
for ln in open('/home/ubuntu/zxy/vlm-memory/RESULTS_GRID.md'):
    if not ln.startswith('|') or '---' in ln or 'panel' in ln:
        continue
    c = [x.strip() for x in ln.strip().strip('|').split('|')]
    if len(c) < 4:
        continue
    if c[0]:
        panel = c[0]
    arm = c[2]
    if arm in TPL:
        rows.append((panel, arm, c[3:3+len(RAT)]))

bad, checked, skipped = [], 0, 0
for panel, arm, cells in rows:
    ds = PANEL.get(panel)
    if ds is None:
        continue
    for r, cell in zip(RAT, cells):
        if cell in ("—", ""):
            continue
        m_tab = float(re.match(r'([+-][\d.]+)', cell.replace('−', '-')).group(1))
        base = read_scores(ds, "_b002" if r == 0.02 else "_g8base", r, strict=False)
        if len(base) < 5:
            skipped += 1
            continue
        if arm in SEEDED:
            ms = []
            for S in (0, 1, 2):
                o = read_scores(ds, TPL[arm].format(S=S), r, strict=False)
                if len(set(o) & set(base)) >= 5 and len(o) == len(base):
                    ms.append(paired(o, base)[0])
            if not ms:
                skipped += 1
                continue
            m_new = float(np.mean(ms))
            n_new = len(ms)
        else:
            o = read_scores(ds, TPL[arm], r, strict=False)
            if len(set(o) & set(base)) < 5:
                skipped += 1
                continue
            m_new = paired(o, base)[0]
            n_new = 1
        checked += 1
        if abs(m_new - m_tab) > 0.02:
            bad.append((panel, arm, r, m_tab, m_new, n_new))
        # 种子数也要对上
        mseed = re.search(r'\((\d)\)', cell)
        if mseed and int(mseed.group(1)) != n_new:
            bad.append((panel, arm, r, f"seed{mseed.group(1)}", f"seed{n_new}", n_new))

print(f"复核 {checked} 格（跳过 {skipped} 格：基线或臂无数据）")
if bad:
    print(f"\n!!! {len(bad)} 格不一致：")
    for b in bad:
        print("   ", b)
else:
    print("全部一致 ✓（|差| ≤ 0.02，种子数也对上）")
