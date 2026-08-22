#!/usr/bin/env python3
"""把 selfstudy 各分片的产出合回 manifest，并跑质量闸。

分片只写自己处理过的记录（避免互相覆盖），这里按 `id` 合并回底稿。
合并后**立刻**跑 `audit_questions`，并打印 sha256 —— 数据集从此可冻结引用。
"""
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prometa.dataset import audit_questions   # noqa: E402


def main():
    base, pat, out = sys.argv[1], sys.argv[2], sys.argv[3]
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(base)}
    n = 0
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            r = json.loads(l)
            if r.get("ss_ok"):
                recs[r["id"]] = r; n += 1
            elif r["id"] in recs and r.get("ss_ok") is False:
                recs[r["id"]]["ss_ok"] = False
    allr = [recs[k] for k in sorted(recs)]
    with open(out, "w") as fo:
        for r in allr:
            fo.write(json.dumps(r) + "\n")
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    usable = [r for r in allr if r["futures"]]
    done = [r for r in allr if r.get("ss_ok")]
    print(f"合并 {n} 条 selfstudy → {out}")
    print(f"  可训练 {len(usable)}/{len(allr)} 条；selfstudy 成功 {len(done)}")
    print(f"  sha256 {h}")
    if done:
        au = audit_questions(done)
        print(f"\n== 质量闸 ==\n  问句数 {au['n_q']}  **grounded {au['grounded']:.3f}**"
              f"（<0.15 ⇒ 多半在编）")
        print("  **同类型跨文档 Jaccard**（>0.6 ⇒ 生成器退化成模板）：")
        for t, v in sorted(au["cross_doc_J"].items()):
            print(f"     {t:<12} {v:.3f}{'  ⚠退化' if v > 0.6 else ''}")
        print(f"     {'均值':<12} {au['cross_doc_J_mean']:.3f}")


if __name__ == "__main__":
    main()
