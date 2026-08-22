#!/usr/bin/env python3
"""**selfstudy 问句生成** —— 让未来真正由文档内容决定（ProMeta 数据集 A 层）。

────────────────────────────────────────────────────────────────────────────
核心设计：**按窗口生成，不是把整篇喂进去**
────────────────────────────────────────────────────────────────────────────
最直接的做法是「整篇 128k 丢给模型，让它出 5 个问题」。**不采用**，三个理由：

① **lost-in-the-middle**：模型多半只从开头/结尾取材 ⇒ 五个问句的证据都挤在两端
   ⇒ 教师的 `U` 也挤在两端，中间几万 token 永远没有未来需要它。那是**位置偏置**，
   会让 Student 学到「留头留尾」这条与内容无关的捷径。
② **贵**：128k 预填 × 122 篇。按窗口只要 5 × 16k。
③ **多未来会退化成一个**：五个问句若都取材同一段，`U_1≈…≈U_5`，
   多未来就是假的（`dataset.audit_footprint` 专门查这条）。

所以按**类型分配取材范围**，让五个未来**按构造**就有不同的 memory footprint：

    factual / method / comparison → 各自一个**随机窗口**（互不重叠，均匀铺开）
    summary                       → **等距抽样**整篇（12 段 × 1000 token）
    multihop                      → **两个相距很远**的窗口拼起来

⚠ 生成器与教师用**同一个冻结 backbone**（RestoreKV 的 self-study 就是如此）。
⚠ 问句只用于**构造未来**；教师算 `U` 时用的仍是**原始 token id 的上下文**，
   与这里 decode→encode 的漂移无关。

────────────────────────────────────────────────────────────────────────────
质量闸（不通过就不该用这批数据）
────────────────────────────────────────────────────────────────────────────
生成完立刻跑 `dataset.audit_questions`：
  · `grounded`     问句实词落在**该文档**里的比例。低 ⇒ 在编。
  · `cross_doc_J`  **同类型跨文档**问句的 Jaccard。**高 ⇒ 生成器退化成模板** ——
    那样未来又不依赖内容了，与合成 shortcut 是同一个病换了件外衣。
自测里的正负对照：内容衍生 J=0.333 / 模板化 **J=1.000**。

    .venv/bin/python scratch_prometa_selfstudy.py \\
        --manifest prometa_data/manifest_200.jsonl --out prometa_data/manifest_200_ss.jsonl
"""
import argparse
import json
import os
import random
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prometa.dataset import QTYPES, audit_questions, parse_selfstudy   # noqa: E402

ONE_Q = (
    "Read the excerpt above.\n"
    "Write exactly ONE question of type {t} about it.\n"
    "Type {t} means: {d}\n"
    "The question must be answerable from the excerpt alone and at most 30 words.\n"
    "It must mention specific content from the excerpt, not generic wording.\n"
    "Output exactly one line in the format {t}|your question — nothing else.\n"
)


# 三类 local 问题各自的取材**带**（占全文比例），**互不重叠**。
# ⚠ 外部复核指出：首版每类独立 `rng.randrange` 取窗，**没有任何互斥约束**，
#    factual 30–46k / method 34–50k / comparison 28–44k 大面积重合完全可能
#    ⇒ 我在文档里写的「按构造保证不同 footprint」**不成立**（第⑤类错：
#    理论声明与实现不一致）。现在按带分配，带内再随机 jitter。
LOCAL_BANDS = {"factual": (0.08, 0.30), "method": (0.36, 0.58),
               "comparison": (0.64, 0.92)}
MULTIHOP_BANDS = ((0.03, 0.20), (0.78, 0.97))


def _slice_frac(ids, f0, f1, win, rng):
    n = len(ids)
    lo, hi = int(f0 * n), int(f1 * n)
    w = min(win, max(1000, hi - lo))
    off = rng.randrange(lo, max(lo + 1, hi - w + 1))
    return ids[off:off + w]


def windows_for(ids, qtype, win, rng):
    """→ 该类型的取材 token 段。**类型决定范围**，这是 footprint 分化的来源。

    · summary   等距抽样整篇（12 段 × 1000 token）
    · multihop  两个**固定在首尾带**的窗口
    · 其余三类  各自一个**专属带**内的随机窗口，**三带互不重叠**（见 LOCAL_BANDS）
    """
    n = len(ids)
    if qtype == "summary":
        k, seg = 12, 1000
        if n <= k * seg:
            return ids
        starts = [int(i * (n - seg) / (k - 1)) for i in range(k)]
        out = []
        for s_ in starts:
            out += ids[s_:s_ + seg]
        return out
    if qtype == "multihop":
        (a0, a1), (b0, b1) = MULTIHOP_BANDS
        half = max(1, win // 2)
        return (_slice_frac(ids, a0, a1, half, rng)
                + _slice_frac(ids, b0, b1, half, rng))
    f0, f1 = LOCAL_BANDS[qtype]
    return _slice_frac(ids, f0, f1, win, rng)


def audit_window_overlap(n=100000, win=16000, seed=0):
    """自检：三类 local 窗口在**任何** jitter 下都不重叠。判据写成代码。"""
    import random as _r
    iv = {}
    for t in LOCAL_BANDS:
        lo, hi = LOCAL_BANDS[t]
        iv[t] = (int(lo * n), int(hi * n))
    ks = list(iv)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = iv[ks[i]], iv[ks[j]]
            assert a[1] <= b[0] or b[1] <= a[0], (ks[i], ks[j], a, b)
    # 实际取到的窗口也必须落在带内
    ids = list(range(n))
    for t in LOCAL_BANDS:
        for sd in range(50):
            seg = windows_for(ids, t, win, _r.Random(sd))
            lo, hi = iv[t]
            assert seg[0] >= lo and seg[-1] < hi + win, (t, sd, seg[0], seg[-1], iv[t])
    return {t: iv[t] for t in iv}


@torch.inference_mode()
def gen_one(model, tok, ctx_ids, qtype, desc, win, rng, max_new=64):
    seg = windows_for(ctx_ids, qtype, win, rng)
    text = tok.decode(seg, skip_special_tokens=True)
    msg = [{"role": "user", "content": text + "\n\n" + ONE_Q.format(t=qtype, d=desc)}]
    ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
    ids = ids.to(model.device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


STOP = set(("what which who whom whose when where why how does do did is are was were "
            "the a an of in on for to from by with and or as at that this these those "
            "it its their his her they he she you your text document passage article "
            "above excerpt described mentioned discussed").split())


def grounded_score(q, ctx_text):
    w = {x for x in re.findall(r"[a-z]{3,}", q.lower())} - STOP
    d = {x for x in re.findall(r"[a-z]{3,}", ctx_text.lower())} - STOP
    return (len(w & d) / len(w)) if w and d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--win", type=int, default=16000, help="单窗口取材长度")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 条（冒烟）")
    ap.add_argument("--retries", type=int, default=2,
                   help="单个类型生成失败（含类型不符）的重试次数")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--audit_only", action="store_true",
                   help="只跑零 GPU 的窗口互斥自检")
    a = ap.parse_args()
    if a.audit_only:
        iv = audit_window_overlap()
        print("三类 local 取材带**互不重叠**（按 10 万 token 折算）：")
        for t, (lo, hi) in iv.items():
            print(f"  {t:<12} [{lo:,}, {hi:,})")
        print("multihop 固定在首尾带", MULTIHOP_BANDS, "；summary 等距抽样整篇")
        print("PASS")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2").eval()
    print(f"[selfstudy] 模型就绪 {a.model}", flush=True)

    recs = [json.loads(l) for l in open(a.manifest)]
    todo = [r for r in recs if r["kind"] == "selfstudy" and not r["futures"]]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[selfstudy] 待生成 {len(todo)} / 共 {len(recs)} 条", flush=True)

    ok, fail = 0, 0
    for n_, r in enumerate(todo):
        rng = random.Random((a.seed, r["id"]).__hash__())
        ctx = r["ctx"]
        ctx_text = tok.decode(ctx[:200000], skip_special_tokens=True)
        futs = []
        for qtype, desc in QTYPES:
            q = None
            for attempt in range(a.retries + 1):
                raw = gen_one(model, tok, ctx, qtype, desc, a.win, rng)
                got = parse_selfstudy(raw, n=1)
                # ⚠ **必须校验类型本身**（外部复核指出，采纳）：`parse_selfstudy`
                #    接受任何合法 QTYPES，若要 factual 而模型输出 `summary|...`，
                #    首版会把它**重新标成 factual** —— 标签与内容不符且**不报错**。
                if got and got[0][0] == qtype and len(got[0][1]) > 8:
                    q = got[0][1]; break
            if q is None:
                continue
            futs.append(dict(kind=qtype, q_text=q,
                             q=tok.encode(f"\nQuestion: {q}\nAnswer:",
                                          add_special_tokens=False),
                             a=[], needs=[],
                             grounded=round(grounded_score(q, ctx_text), 4)))
        # **硬闸：必须 5/5**（外部复核指出，采纳）。旧的 `>=4` 会产出 M=4 的记录，
        # 而 Student 那边是 `assert len(futures) == Mt`（Mt=5 固定 probe 数）——
        # 接进训练时会直接冲突。为几条失败样本把训练代码变复杂不值得，
        # 缺一类就重试（`--retries`），仍失败就丢掉这条 context。
        if len(futs) == len(QTYPES):
            r["futures"] = futs
            r["ss_ok"] = True
            ok += 1
        else:
            r["ss_ok"] = False
            fail += 1
        if (n_ + 1) % 10 == 0 or n_ + 1 == len(todo):
            print(f"[selfstudy] {n_+1}/{len(todo)}  成功 {ok} 失败 {fail}", flush=True)

    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"\n写出 {len(recs)} 条 → {a.out}（selfstudy 成功 {ok} / 失败 {fail}）")

    # ── 质量闸 ────────────────────────────────────────────────────────────
    done = [r for r in recs if r.get("ss_ok")]
    gr = [f["grounded"] for r in done for f in r["futures"]]
    au = audit_questions(done)
    import numpy as np
    print("\n== 问句质量审计（不通过就不该用这批数据）==")
    print(f"  问句数 {len(gr)}；**grounded 均值 {np.mean(gr):.3f}**"
          f"（中位 {np.median(gr):.3f}，<0.15 ⇒ 多半在编）")
    print(f"  **同类型跨文档 Jaccard**（>0.6 ⇒ 生成器退化成模板）：")
    for t, v in sorted(au["cross_doc_J"].items()):
        flag = "  ⚠退化" if v > 0.6 else ""
        print(f"     {t:<12} {v:.3f}{flag}")
    print(f"     {'均值':<12} {au['cross_doc_J_mean']:.3f}")
    print("\n⚠ 下一步是 footprint 审计（`dataset.audit_footprint`）——它要教师的 U，"
          "必须在抽完 U 之后跑：五问的保留集两两 Jaccard ≈1 ⇒ 多未来是假的。")
    print("Finished.")


if __name__ == "__main__":
    main()
