"""ProMeta 的**正式训练集构建器** —— 三层未来 + 文档级划分 + 长度分层。

────────────────────────────────────────────────────────────────────────────
为什么需要它（一句话）
────────────────────────────────────────────────────────────────────────────
Student 学的是 `P(未来需求 | 上下文)`。要让这个条件分布**真的依赖上下文**，
未来就必须**由文档自身内容决定**。现有的 `teacher.build_task` 把同一套模板事实
（`The secret key {k} maps to the value {v}.`）插进每一篇文档 ⇒
**换一篇文档，未来需求几乎不变** ⇒ 那样的语料扩到 4,000 篇也只是同一道题做 4,000 遍。

⚠ **这仍是待验证的假说，不是结论**（外部复核的修正，采纳）：key/value、插入位置、
三个事实的身份、joint 组合都在变，Student 要**分别**预测 `U_1..U_5` 未必能靠一组
常数 probe 做到。`--no_context` 致盲对照与 `--shuffle_labels` 打乱对照就是判它的。

────────────────────────────────────────────────────────────────────────────
三层未来（比例是建议值，由 `--mix` 控制）
────────────────────────────────────────────────────────────────────────────
A. `selfstudy` **60–70%**：冻结的 backbone 读文档、**按文档内容**生成 5 类异质问题
   （factual / summary / multihop / method / comparison）。这是 RestoreKV 的
   self-study（arXiv 2608.01247，已读原文：LongAlpaca 500 篇每篇 5 类问题，
   ≤64 token）。**它是让未来真正绑定内容的唯一来源。**
   ⚠ 关键不在「五个名字」，而在它们要有**不同的 memory footprint**：
   factual 需要极少量局部 token、summary 需要大范围分布证据、multihop 需要
   两处远距离证据 —— 否则五个问句都指向同一段，`U_1≈…≈U_5`，等于没有多未来。
   `audit_footprint` 就是查这一条的。
B. `synth` **20–30%**：现有 `build_task`。**不要扔** —— 它的 private / joint /
   全共享结构是**已知的 ground truth**，是唯一能做机制消融（风险维度、
   命中率归因）的数据。定位是 **controlled curriculum / diagnostic**，不是主体。
C. `continuation` **10–20%**：把文档自己的后续 `x_{T+1:T+H}` 当成一个真实未来
   （外部复核提出，采纳）。**它不经过任何 QA 生成器** ⇒ 天然免疫「所有未来都来自
   同一个问答模板」这个风险，也最接近 LookaheadKV 的训练思想
   （arXiv 2603.10899：监督就是 response Y 的 query 对 prompt X 的注意力）。

────────────────────────────────────────────────────────────────────────────
划分与分层
────────────────────────────────────────────────────────────────────────────
· **按文档划分**，同一 context 的所有未来必须进同一 split（训练单位就是
  `(C, {q_m})`，按 query 划分会泄漏）。默认 80/10/10，**按位置切、不 shuffle**
  （`--split_seed` 那个坑：篇数一变划分全变）。
· 长度分层默认 `8-16k:0.35, 16-32k:0.30, 32-64k:0.20, 64-128k:0.15` ——
  两个竞争者都训练在 ≤16K 而在 100K+ 上评测；保留 15% 长样本是为了对齐
  部署端**池化摘要**的输入分布（真正随长度变的只有它）。
  ⚠ 32k 以上的文档在单个 FineWeb 分片里很少（30–60k 只有 278 篇、60–100k 46 篇），
  不够时由短文**拼接**补足，并在 manifest 里标 `built="concat"`。

⚠ **Teacher 计算才是真瓶颈**（外部复核的修正，采纳）：原始语料 4,328 篇是免费的，
但每篇还要满缓存预填 + M 次未来前向 + 算 `U*`。所以先小规模判通道，再放量。
"""
import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "external/FastKVzip/prefill"))

# 五类异质未来。**类型名不是重点，footprint 才是** —— 见模块 docstring A 段。
QTYPES = [
    ("factual", "one specific fact, number, name or definition stated in the text"),
    ("summary", "the overall argument or main contribution of the whole text"),
    ("multihop", "something that requires combining evidence from two DISTANT parts"),
    ("method", "why or how something described in the text works"),
    ("comparison", "a comparison or contrast between two things discussed in the text"),
]

SELFSTUDY_PROMPT = (
    "Read the document above.\n"
    "Write exactly {n} questions about it, one per line, in the format TYPE|question.\n"
    "Use each of these TYPEs exactly once, in this order: {types}.\n"
    "Each question must be answerable from the document alone and at most 30 words.\n"
    "Ask about {hint}\n"
    "Output only the {n} lines, nothing else.\n"
)

BANDS = [(8000, 16000), (16000, 32000), (32000, 64000), (64000, 128000)]


def parse_mix(s_):
    """`"selfstudy:0.65,synth:0.25,continuation:0.10"` → 归一化后的 dict。"""
    d = {}
    for part in s_.split(","):
        k, v = part.split(":")
        d[k.strip()] = float(v)
    tot = sum(d.values())
    assert tot > 0, s_
    return {k: v / tot for k, v in d.items()}


def parse_bands(s_):
    """`"8-16k:0.35,..."` → [(lo, hi, w)]，权重归一化。"""
    out = []
    for part in s_.split(","):
        rng, w = part.split(":")
        lo, hi = rng.replace("k", "").split("-")
        out.append((int(lo) * 1000, int(hi) * 1000, float(w)))
    tot = sum(w for _, _, w in out)
    return [(lo, hi, w / tot) for lo, hi, w in out]


def make_contexts(n_docs, bands, rng_seed, enc, pool_skip=68):
    """→ [(token_ids, band_idx, built)]。长度不够的 band 用短文**拼接**补足。"""
    from prometa.teacher import load_fineweb_pool
    rng = random.Random((rng_seed, "ctx").__hash__())
    ws = np.array([w for _, _, w in bands])
    cum = np.cumsum(ws)
    # 一次性取足够多的短文（10k–30k 那个 band 有 4,328 篇）
    need_pool = max(200, n_docs * 4)
    raw = load_fineweb_pool(need_pool, 10000, 30000, skip=pool_skip)
    toks = [enc(t) for t in raw]
    out, seen = [], set()
    for i in range(n_docs):
        r = random.Random((rng_seed, i, "doc").__hash__())
        bi = int(np.searchsorted(cum, r.random()))
        lo, hi, _ = bands[bi]
        tgt = r.randrange(lo, hi)
        order = list(range(len(toks)))
        r.shuffle(order)
        buf, used = [], 0
        for j in order:
            buf += toks[j]; used += 1
            if len(buf) >= tgt + max(2000, tgt // 4):
                break
        off = r.randrange(0, max(1, len(buf) - tgt + 1))
        ctx = buf[off:off + tgt]
        key = (len(ctx), tuple(ctx[:48]), tuple(ctx[-48:]))
        assert key not in seen, f"第 {i} 条上下文与已有重复"
        seen.add(key)
        out.append((ctx, bi, "single" if used == 1 else "concat"))
    return out


def continuation_future(ids, horizon):
    """→ (前缀, 未来 token)。**前缀必须排除被留作未来的那一段**，否则未来在前缀里。"""
    assert len(ids) > horizon + 1000, (len(ids), horizon)
    return ids[:-horizon], ids[-horizon:]


def build_manifest(a, enc):
    """无 GPU 部分：语料 + 划分 + `synth` / `continuation` 两类未来。

    `selfstudy` 的问句留空，由 `add_selfstudy`（需要 GPU）填上。
    """
    from prometa.teacher import build_task
    bands = parse_bands(a.bands)
    mix = parse_mix(a.mix)
    ctxs = make_contexts(a.n_docs, bands, a.corpus_seed, enc, a.pool_skip)
    n_tr = int(round(a.n_docs * a.frac_train))
    n_va = int(round(a.n_docs * a.frac_val))
    recs = []
    for i, (ids, bi, built) in enumerate(ctxs):
        r = random.Random((a.corpus_seed, i, "fut").__hash__())
        kind = _pick(mix, r)
        split = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
        rec = dict(id=i, split=split, band=f"{bands[bi][0]//1000}-{bands[bi][1]//1000}k",
                   built=built, kind=kind, n_ctx=len(ids))
        if kind == "synth":
            ctx, fut, meta = build_task(enc, ids, a.max_ctx, a.window,
                                        a.n_fact, r, n_joint=a.n_joint)
            rec["ctx"] = ctx
            rec["futures"] = [dict(kind=f["kind"], q=f["q"], a=f["a"],
                                   needs=f["needs"]) for f in fut]
            rec["meta"] = {k: v for k, v in meta.items() if k != "span_len"}
        elif kind == "continuation":
            pre, cont = continuation_future(ids, a.horizon)
            rec["ctx"] = pre
            rec["n_ctx"] = len(pre)
            # 未来 = 文档自己的后续；**没有问句**，整段续写就是 query
            rec["futures"] = [dict(kind="continuation", q=cont, a=[], needs=[])]
        else:                                   # selfstudy：问句待填
            rec["ctx"] = ids
            rec["futures"] = []                 # add_selfstudy 填
        recs.append(rec)
    return recs


def _pick(mix, r):
    x, acc = r.random(), 0.0
    for k, v in mix.items():
        acc += v
        if x <= acc:
            return k
    return list(mix)[-1]


def selfstudy_prompt(n=5):
    types = ", ".join(t for t, _ in QTYPES[:n])
    hint = "; ".join(f"{t}: {d}" for t, d in QTYPES[:n])
    return SELFSTUDY_PROMPT.format(n=n, types=types, hint=hint)


def parse_selfstudy(text, n=5):
    """→ [(type, question)]，容错解析。**解析失败返回空**，调用方必须处理。"""
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("0123456789.)- ")
        if "|" not in line:
            continue
        t, q = line.split("|", 1)
        t, q = t.strip().lower(), q.strip()
        if t in dict(QTYPES) and len(q) > 8:
            out.append((t, q))
    seen, uniq = set(), []
    for t, q in out:
        if t not in seen:
            seen.add(t); uniq.append((t, q))
    return uniq[:n]


def audit_footprint(U, thresh=0.5):
    """**未来 footprint 审计**：五个问句是不是指向同一段？

    `U`: [M, L, H, N]。返回未来两两之间保留集的 Jaccard 均值（同预算 top-k）。
    **≈1 ⇒ 五个问句要的是同一批位置 ⇒ 多未来是假的**，这时类型名再漂亮也没用。
    """
    from prometa.risk import topb_mask
    U = np.asarray(U, dtype=np.float64)
    M, L, H, N = U.shape
    k = max(1, int(round(0.02 * N)))
    ms = [topb_mask(U[m], k) for m in range(M)]
    js = []
    for i in range(M):
        for j in range(i + 1, M):
            a_, b_ = ms[i], ms[j]
            js.append(((a_ & b_).sum(-1) / np.maximum((a_ | b_).sum(-1), 1)).mean())
    return float(np.mean(js))


def audit_questions(recs, enc=None):
    """**问句质量审计** —— 这才是 selfstudy 值不值钱的判据，不是「生成成功了几条」。

    两个量，都必须报：

    · `grounded`  问句里的实词有多大比例真的出现在**该文档**里。
      低 ⇒ 生成器在编，或者只会问通用问题。
    · `cross_doc_J` **同类型、跨文档**的问句 token 集合 Jaccard 均值。
      **高 ⇒ 生成器退化成模板**（例如每篇的 summary 问句都是
      「What is the main argument of this text?」）—— 那样未来又不依赖内容了，
      与合成 shortcut 是同一个病，只是换了个外衣。
      **这是这一层数据唯一真正的失败模式，必须查。**
    """
    import re
    STOP = set(("what which who whom whose when where why how does do did is are was "
                "were the a an of in on for to from by with and or as at that this "
                "these those it its their his her they he she you your text document "
                "passage article above described mentioned discussed").split())
    def words(t):
        return {w for w in re.findall(r"[a-z]{3,}", t.lower())} - STOP
    per_type, gr = {}, []
    for r in recs:
        if r.get("kind") != "selfstudy" or not r.get("futures"):
            continue
        dw = words(r.get("ctx_text", ""))
        for f in r["futures"]:
            qt = f.get("q_text", "")
            w = words(qt)
            if "grounded" in f:
                # 生成时已算好（上下文可达 12 万 token，**不存进 manifest**）
                gr.append(float(f["grounded"]))
            elif dw and w:
                gr.append(len(w & dw) / len(w))
            per_type.setdefault(f["kind"], []).append(w)
    out = dict(n_q=len(gr), grounded=float(np.mean(gr)) if gr else float("nan"))
    js = {}
    for t, ws in per_type.items():
        pair = []
        for i in range(len(ws)):
            for j in range(i + 1, min(len(ws), i + 12)):     # 抽样，别 O(n²) 全算
                u = ws[i] | ws[j]
                pair.append(len(ws[i] & ws[j]) / len(u) if u else 0.0)
        js[t] = float(np.mean(pair)) if pair else float("nan")
    out["cross_doc_J"] = js
    out["cross_doc_J_mean"] = float(np.nanmean(list(js.values()))) if js else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=200)
    ap.add_argument("--mix", default="selfstudy:0.65,synth:0.25,continuation:0.10")
    ap.add_argument("--bands", default="8-16k:0.35,16-32k:0.30,32-64k:0.20,64-128k:0.15")
    ap.add_argument("--frac_train", type=float, default=0.8)
    ap.add_argument("--frac_val", type=float, default=0.1)
    ap.add_argument("--corpus_seed", type=int, default=20260822)
    ap.add_argument("--pool_skip", type=int, default=68)
    ap.add_argument("--max_ctx", type=int, default=128000)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--n_fact", type=int, default=3)
    ap.add_argument("--n_joint", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--out", default="prometa_data/manifest.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    enc = lambda t: tok.encode(t, add_special_tokens=False)      # noqa: E731
    recs = build_manifest(a, enc)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    ck, cb, cs = Counter(), Counter(), Counter()
    for r in recs:
        ck[r["kind"]] += 1; cb[r["band"]] += 1; cs[r["split"]] += 1
    print(f"写出 {len(recs)} 条 → {a.out}")
    print(f"  未来来源 {dict(ck)}")
    print(f"  长度分层 {dict(cb)}")
    print(f"  划分     {dict(cs)}（**按文档**，同一 context 的未来不跨 split）")
    print(f"  上下文 token 合计 {sum(r['n_ctx'] for r in recs):,}")
    ns = sum(1 for r in recs if r["kind"] == "selfstudy")
    print(f"  ⚠ 其中 {ns} 条的 selfstudy 问句**尚未生成**，需要 GPU 跑 `add_selfstudy`")


def _selftest():
    """零 GPU 自测：只用一个假 tokenizer 验结构、划分、去重、footprint 判据。"""
    class FakeEnc:
        def __call__(self, s_):
            return [abs(hash(w)) % 50000 for w in s_.split()]
    # ① mix / bands 解析归一
    m = parse_mix("selfstudy:0.65,synth:0.25,continuation:0.10")
    assert abs(sum(m.values()) - 1) < 1e-9 and len(m) == 3, m
    b = parse_bands("8-16k:0.35,16-32k:0.30,32-64k:0.20,64-128k:0.15")
    assert abs(sum(w for _, _, w in b) - 1) < 1e-9 and b[0][:2] == (8000, 16000), b
    print(f"① mix/bands 解析归一　PASS　{m}")

    # ② continuation：前缀必须**不含**被留作未来的那一段
    ids = list(range(5000))
    pre, cont = continuation_future(ids, 256)
    assert len(pre) == 4744 and cont == list(range(4744, 5000))
    assert not set(pre) & set(cont), "前缀里混进了未来 token"
    print(f"② continuation 前缀 {len(pre)} / 未来 {len(cont)}，无重叠　PASS")

    # ③ selfstudy 解析：容错 + 去重 + 只收合法类型
    txt = ("1) factual|What dimensionality was used?\n"
           "junk line without a bar\n"
           "- summary|What is the main contribution?\n"
           "FACTUAL|duplicate type should be dropped\n"
           "multihop|How does Sec 2 relate to Sec 5?\n"
           "bogus|not a real type\n"
           "method|Why does it reduce computation?\n"
           "comparison|Compare A and B.\n")
    got = parse_selfstudy(txt)
    assert [t for t, _ in got] == ["factual", "summary", "multihop", "method",
                                   "comparison"], got
    print(f"③ selfstudy 解析出 {len(got)} 类、去重且拒非法类型　PASS")

    # ④ footprint 判据要能区分「五问指向同一段」与「各指一段」
    rs = np.random.default_rng(0)
    N = 500
    same = np.tile((rs.random((1, 2, 2, N)) ** 6), (5, 1, 1, 1))
    diff = rs.random((5, 2, 2, N)) ** 6
    for m_ in range(5):
        diff[m_, :, :, m_ * 80:(m_ + 1) * 80] += 3.0
    j_same, j_diff = audit_footprint(same), audit_footprint(diff)
    assert j_same > 0.9 and j_diff < 0.2, (j_same, j_diff)
    print(f"④ footprint 判据：五问同段 J={j_same:.3f}、各指一段 J={j_diff:.3f}　PASS")

    # ⑤ 问句审计判据的正负对照：模板化生成器必须被抓出来
    # ⚠ **夹具的区分位必须对度量可见**（同一失效模式本会话已犯三次）：首版用
    #    `delta0/echo0` 编号区分，而 `[a-z]{3,}` 只取字母、数字被剥掉 ⇒ 两边都变成
    #    `{delta, echo}`，J 平凡地等于 1，看上去像判据失灵。改用真正不同的实词。
    TOPIC = ["photosynthesis chloroplast", "renaissance florence", "quicksort pivot",
             "insulin pancreas", "sonnet iambic", "titration burette",
             "monsoon precipitation", "blockchain consensus", "myelin axon",
             "tariff embargo", "sediment stratigraphy", "vaccine adjuvant"]
    good = [dict(kind="selfstudy",
                 ctx_text=f"introduction {TOPIC[i]} conclusion",
                 futures=[dict(kind="summary",
                               q_text=f"What role does {TOPIC[i]} serve?")])
            for i in range(12)]
    bad = [dict(kind="selfstudy",
                ctx_text=f"introduction {TOPIC[i]} conclusion",
                futures=[dict(kind="summary", q_text="What is the main argument here?")])
           for i in range(12)]
    ag, ab = audit_questions(good), audit_questions(bad)
    assert ag["cross_doc_J_mean"] < 0.35, ag
    assert ab["cross_doc_J_mean"] > 0.9, ab
    assert ag["grounded"] > ab["grounded"], (ag["grounded"], ab["grounded"])
    print(f"⑤ 问句审计：内容衍生 J={ag['cross_doc_J_mean']:.3f} grounded={ag['grounded']:.2f}；"
          f"**模板化 J={ab['cross_doc_J_mean']:.3f} grounded={ab['grounded']:.2f}** ⇒ 判据能抓出退化　PASS")

    # ⑥ 划分按文档、同一 context 的未来不跨 split（结构性保证：futures 挂在 rec 上）
    a = argparse.Namespace(n_docs=20, mix="synth:0.5,continuation:0.5",
                           bands="8-16k:1.0", frac_train=0.8, frac_val=0.1,
                           corpus_seed=1, pool_skip=68, max_ctx=128000,
                           window=4096, n_fact=3, n_joint=2, horizon=256)
    print("⑥ 划分/去重需要真语料，见 `--n_docs 20` 的实跑（本自测不下载数据）")
    print("\nprometa/dataset.py 自测 5 条（零 GPU 部分）全过")


if __name__ == "__main__":
    main()
