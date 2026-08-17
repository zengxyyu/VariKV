#!/usr/bin/env python3
"""拆解四臂的下游配对报表 —— +4.27 到底来自校准还是 token 内容。

**参照必须用 v3 而不是 v2**（CLAUDE.md 的"训练篇数陷阱"）：四臂与 v3 同为
23/7 划分、同 637,828 参数、同 memory 架构；v2 是 8/2，差 3 倍训练数据，
拿它当分母会把架构差异和数据量混在一起。

口径沿用项目惯例：只报**配对 Δ**（相对同一批 `__g8base`），逐样本 bootstrap，
★ = 95% CI 不含 0。绝对分只在两臂样本集完全一致时才可比。
"""
import contextlib, io, json, os, sys
import numpy as np
_P = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "external/FastKVzip/prefill")
sys.path.insert(0, _P); os.chdir(_P)
from results.parse import parse_answer, evaluate_answer            # noqa: E402

DATA, RATIO = "scbench_kv", 0.1
ARMS = [("baseline", "__g8base_chunk16k_w4096", "—"),
        ("bias 225", "__dcbias_chunk16k_w4096_ctrlmstat8_bias", "纯预算再分配"),
        ("affine 225", "__dcaffine_chunk16k_w4096_ctrlmstat8_affine", "+尺度重校准"),
        ("scalar 4.5K", "__dcscalar_chunk16k_w4096_ctrlmstat8_scalar", "只看分数统计量"),
        ("kv 53K", "__dckv_chunk16k_w4096_ctrlmstat8_kv", "只看 KV 内容"),
        ("v3 638K★参照", "__g8v3_chunk16k_w4096_ctrlmmemo8", "同 23/7 划分的 full"),
        ("v2 638K", "__g8v2_chunk16k_w4096_ctrlmmemo8", "8/2 划分，仅供参考")]
_M = contextlib.redirect_stdout(io.StringIO())


def per_sample(sfx):
    with _M:
        ANSW, SUBT = parse_answer(DATA)
    out, i = {}, 0
    while True:
        f = f"results/{DATA}/{i}_qwen2.5-7b-instruct-1m_fastkvzip{sfx}/output-pair.json"
        if not os.path.exists(f):
            break
        dd = json.load(open(f)); p, ans = [], []
        for fmt in [k for k in dd if k.startswith("qa")]:
            for info, txt in dd[fmt]:
                if abs(float(info[0]) - RATIO) < 1e-9:
                    p.append(txt["pruned"]); ans.append(txt["answer"])
        if p:
            with _M:
                out[i] = float(np.mean(evaluate_answer(
                    p, ANSW[i] if ANSW else ans, DATA, "qa",
                    subtask=SUBT[i] if SUBT else None)))
        i += 1
    return out


def boot(d, n=4000, seed=0):
    r = np.random.default_rng(seed)
    b = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


B = per_sample(ARMS[0][1])
print(f"{DATA} @ ratio {RATIO}　基线 n={len(B)}　绝对分 {np.mean(list(B.values()))*100:.2f}")
print(f"\n{'臂':<15}{'n':>5}{'配对 Δ':>12}{'95% CI':>20}  机制")
for name, sfx, note in ARMS[1:]:
    A = per_sample(sfx)
    c = sorted(set(A) & set(B))
    if not c:
        print(f"{name:<15}{'—':>5}  缺数据"); continue
    d = (np.array([A[j] for j in c]) - np.array([B[j] for j in c])) * 100
    m, lo, hi = boot(d)
    star = "★" if (lo > 0 or hi < 0) else ""
    print(f"{name:<15}{len(c):>5}{m:>+11.2f}{star}  [{lo:+6.2f},{hi:+6.2f}]  {note}")
print("\n判读（预注册）：bias/affine ≈ v3 ⇒ 增益是跨层头预算再校准，方法可缩到 225 参数；"
      "\n              kv ≫ bias/affine ⇒ KV 内容确有 gate 没抓到的信息，原故事成立")
