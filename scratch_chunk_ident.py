"""chunk 模式标签能不能**直接反解出 u**？—— 这是「不可学」结论的前提检验。

区分两件事：
  (a) 标签本身有没有可反解的线性成分（`A ≈ uᵀd`）—— 本脚本测这个；
  (b) 能不能从推理时可见的特征**预测**那个 u —— 十之七测的是这个。
若 (a) 都不成立，(b) 的零结果就不能读成「特征没用」，只能读成「靶子没信号」。
样本量沿 5 / 10 / 20 / 30 篇拾级而上，看 R² 是否随 n 抬头。
"""
import json
import sys
import numpy as np
sys.path.insert(0, "/home/ubuntu/zxy/vlm-memory")
from scratch_u_to_table import fit_u                      # noqa: E402

FILES = ["scratch_labm_d100.json", "scratch_labm_d110.json", "scratch_labm_d120.json"]
recs = []
for f in FILES:
    r = json.load(open(f))
    off = len(recs)
    for x in r:
        y = dict(x)
        y["doc"] = f"{f[-8:-5]}:{x['doc']}"      # 三个文件的 doc 编号会撞，加前缀
        recs.append(y)
    print(f"{f}: {len(r)} 条, {len({x['doc'] for x in r})} 篇 (累计 {len(recs)})  off={off}")
docs = sorted({x["doc"] for x in recs})
print(f"合计 {len(recs)} 条 / {len(docs)} 篇\n")

print(f"{'篇数':>4s} {'n':>6s} {'R²':>9s} {'λ':>7s} {'‖u‖₂':>9s}   （R² 是 5 折 CV，负值 = 不如常数）")
for nd in (5, 10, 20, len(docs)):
    sub = [x for x in recs if x["doc"] in set(docs[:nd])]
    u, r2, lam = fit_u(sub)
    print(f"{nd:>4d} {len(sub):>6d} {r2:>+9.4f} {lam:>7g} {np.linalg.norm(u):>9.5f}")

# 参照：bulk 模式（造表用的那份），同样口径
b = json.load(open("scratch_adv_grad_bulk.json"))
u, r2, lam = fit_u(b)
print(f"\n参照 bulk(grad) {len(b):>6d} 条 / {len({x['doc'] for x in b})} 篇："
      f" R²={r2:+.4f} λ={lam:g} ‖u‖₂={np.linalg.norm(u):.5f}")
print("\n判词：")
print("  · bulk R² 明显为正而 chunk 在任何样本量下都 ≈0/为负 ⇒ 是 **chunk 这个标签形态**")
print("    没有可反解的线性成分，不是样本量不够；十之七的「不可学」在 chunk 支上")
print("    应改写为「靶子本身未被识别」。")
print("  · 若 chunk R² 随篇数抬头 ⇒ 反过来，是样本量问题，需补篇。")
