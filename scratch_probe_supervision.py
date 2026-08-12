"""短文档 vs 长文档：谁贡献了有效监督？（免训练，纯分析训练日志 + 一次统计）

起因：我说过「26/30 个记录点 gap 相同 ⇒ 87% 数据相同 ⇒ 分歧不来自数据」。
GPT 指出这个论据太松：不同的那 13% 恰好是 5 篇 103k–122k 的长文档，而训练用
**eviction-sensitive 加权** `w_t ∝ KL(p_F‖p_P)`，所以一个样本对梯度的重要性不是
1/34，而取决于它有多少高 gap 的位置。若长文档贡献了大部分有效监督，那
「87% 相同」就完全不能支撑「数据不是原因」。

本脚本从训练日志里按 (文档是短还是长) 拆开统计 gap 与梯度范数。
判据：长文档若只占 15% 的步数却贡献 >40% 的 Σgap 或 Σ|∇|，则我的论据作废。
"""
import re, sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"external/FastKVzip/prefill"))
LOG = ROOT/"scratch_stage2b_logs/train_kl_v2b_dist.log"
# v2b 的计划是确定性的：pool[k % 34]，前 29 篇短、后 5 篇长（n_short=29,n_long=5）
rows=[]
for ln in open(LOG):
    m=re.match(r"^step +(\d+) .*gap ([\d.]+) resid ([\d.]+).*\|g\|([\d.e+-]+)",ln)
    if not m:
        m2=re.match(r"^step +(\d+) kl [\d.]+ \|g\|([\d.e+-]+).*gap ([\d.]+) resid ([\d.]+)",ln)
        if m2: rows.append((int(m2.group(1)),float(m2.group(3)),float(m2.group(4)),float(m2.group(2))))
        continue
    rows.append((int(m.group(1)),float(m.group(2)),float(m.group(3)),float(m.group(4))))
A=np.array(rows)
if not len(A): sys.exit("日志里没解析到记录")
is_long = (A[:,0] % 34) >= 29          # 计划 di = pool[k % 34]
print(f"解析到 {len(A)} 个记录点（每 50 步一次）")
print(f"  其中长文档步数 {int(is_long.sum())}（{100*is_long.mean():.0f}%）\n")
print(f"{'':<14}{'步数占比':>10}{'Σgap 占比':>12}{'Σ|∇| 占比':>12}{'gap 均值':>10}{'|∇| 均值':>11}")
for nm,sel in (("短文档",~is_long),("**长文档**",is_long)):
    if sel.sum()==0: continue
    print(f"{nm:<14}{100*sel.mean():>9.0f}%{100*A[sel,1].sum()/A[:,1].sum():>11.0f}%"
          f"{100*A[sel,3].sum()/A[:,3].sum():>11.0f}%{A[sel,1].mean():>10.4f}{A[sel,3].mean():>11.2e}")
print("\n判据：长文档占 ~15% 步数却贡献 >40% 的 Σgap 或 Σ|∇| ⇒ 「87% 数据相同」作废")
