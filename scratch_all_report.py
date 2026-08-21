#!/usr/bin/env python3
"""把学习残差（v2/v3）与训练无关质心（K=16/K=1024）放进同一张表。

统一口径，三条都是这个项目付过学费的：

1. **只报配对 Δ，不报跨运行的绝对分。** `results.parse` 的相对行按各自的满缓存分
   归一，而满缓存分逐运行漂移；绝对分也只在两臂样本集完全一致时才可比（慢的那一臂
   没跑到的样本会被交集丢掉）。
2. **共同基线**用 `__g8base`（全 11 panel × 8 ratio 的那次），这样质心与残差对的是
   同一批基线数字，两条线之间才可比。
3. **完成判定看日志的 `Finished.`，不看结果文件计数** —— choice_eng 18 条、
   qa_eng 20、many_shot 54、repoqa 88、vt 90，计数法会把完整的当成截断的。

`--md` 输出 markdown。
"""
import argparse
import contextlib
import io
import functools
import json
import re
import os
import sys

import numpy as np

_P = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "external/FastKVzip/prefill")
sys.path.insert(0, _P)
os.chdir(_P)
from results.parse import parse_answer, evaluate_answer          # noqa: E402

RAT = [1.0, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]
# 基线在 0.02 上是另一批（`__b002`），因为 `__g8base` 只跑了 8 个 ratio
BASE_SFX = {0.02: "__b002_chunk16k_w4096"}
_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch_ctrl_logs")
# `ratio×clen < window` ⇒ chunk_ratio 置 0、只留最近的 token、门控分数不参与
# ⇒ 任何改分数的方法恒为 no-op。这些格标 `°`，Δ 应为 0，非零即实现有问题。
TOK = {"gsm": 86, "squad": 203, "scbench_many_shot": 26474, "scbench_repoqa": 72499,
       "scbench_prefix_suffix": 112635, "scbench_summary": 117806,
       "scbench_choice_eng": 119299, "scbench_qa_eng": 122101,
       "scbench_vt": 124551, "scbench_mf": 149860, "scbench_kv": 169428}



_DEGEN_CACHE = {}


def _degen_measured(ds):
    """从评测日志读 runtime **实测**的退化标记，胜过用标注长度套公式。

    `model/wrapper.py` 每个样本打一行
    `[effective] clen=.. window=.. chunk_ratio=.. degenerate=..`。
    退化时 `chunk_ratio` 归零、`window` 被改写成 `int(ratio*clen)`，
    所以**名义 ratio 可由 `window/clen` 反解**；非退化时 window 保持 4096，
    名义 ratio 由 `chunk_ratio` 近似（两者在非退化格上都不影响判定）。

    这样做的理由：公式回退用的是 `TOK` 里的**标注**长度，实测可差 8.7%，
    于是 repoqa@0.05 这类格落进 [0.8,1.25] 的"判不了"带只能标 `?`。
    而日志里写着确切答案。返回 [(名义 ratio, 是否退化)] 或 None。
    """
    if ds in _DEGEN_CACHE:
        return _DEGEN_CACHE[ds]
    out = None
    for cand in (f"scratch_ctrl_logs/sc11_{ds}_s0.log",
                 f"scratch_ctrl_logs/sc11_{ds}_s1.log"):
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), cand)
        if not os.path.exists(f):
            continue
        seen = {}
        pat = re.compile(r"clen=(\d+) window=(\d+) chunk_ratio=([0-9.]+) "
                         r"degenerate=(True|False)")
        with open(f, errors="ignore") as fh:
            for line in fh:
                mm = pat.search(line)
                if not mm:
                    continue
                clen, win, cr, dg = (int(mm.group(1)), int(mm.group(2)),
                                     float(mm.group(3)), mm.group(4) == "True")
                # **名义 ratio 的反解**：非退化时 `chunk_ratio` 是窗口重标定后的
                # **有效**值（ρ=0.1 在 repoqa 上记成 0.0401），直接拿它当名义 ratio
                # 会让 0.04 这个键在容差内抢先匹配 0.05，把退化格误判成正常。
                #   有效: cr = (ρ·clen − w)/(clen − w)  ⇒  ρ = (cr·(clen−w) + w)/clen
                # 退化时 chunk_ratio 归零、window 被改写成 int(ρ·clen) ⇒ ρ = w/clen。
                r_nom = win / clen if dg else (cr * (clen - win) + win) / clen
                # **记占比而非布尔**：同一 panel 各样本长度不一，同一名义 ratio 下
                # 可能只有一部分样本退化（ManyShot@0.2 即如此：标 ° 却有非零 Δ，
                # 因为非退化的那部分样本仍在真实驱逐）。二值标记会误导。
                key = round(r_nom, 3)
                a, b = seen.get(key, (0, 0))
                seen[key] = (a + (1 if dg else 0), b + 1)
        if seen:
            out = sorted((k, a / b) for k, (a, b) in seen.items())
            break
    _DEGEN_CACHE[ds] = out
    return out

def _degenerate(ds, r):
    """判结构性退化。**优先读 runtime 落盘的实测值，公式只是回退。**

    真实逻辑在 `model/wrapper.py`，两步：

        if clen < prefill_chunk_size:            # 16000
            window_size = int(window_ratio*clen) # window_ratio 默认 0.02
        if chunk_ratio*clen < window_size:
            chunk_ratio = 0.0                    # ⇒ 退化：旧 token 名额为 0，
                                                 #   任何分数扰动都是 no-op

    先前这里写死 `4096/r`、漏掉第一步，把 `gsm`(86 token，真实窗口 1) 与
    `squad`(203，窗口 4) 每一格都误标成退化 —— 于是出现 `+4.00★°` 这种自相矛盾格。

    公式回退还有第二个弱点：`TOK` 是**标注**长度不是实测。反解 `scbench_mf` 的实测
    effective ratio 得 clen≈136,890，而标注是 149,860（差 8.7%）。裕度
    `ρ·clen/W` 落在 [0.8, 1.25] 的格子因此**判定不可靠**，返回 None 表示"未知"，
    由调用方标 `?` 而不是硬判。实测受影响的只有 3 格：
    squad@0.02（裕度 1.015，几乎正好卡在边界）、repoqa@0.05（0.885）、kv@0.02（0.827）。

    返回 True / False / None（未知）。
    """
    m = _degen_measured(ds)
    if m is not None:
        for r_nom, frac in m:
            if abs(r_nom - r) < 0.004:          # 反解后精度很高，容差收紧到 0.4%
                if frac >= 0.9:
                    return True                 # ° 全格退化
                if frac > 0.02:
                    return frac                 # ~ 部分退化，返回占比
                return False
    clen = TOK.get(ds)
    if clen is None:
        return False
    w = 4096 if clen >= 16000 else int(0.02 * clen)
    margin = r * clen / w
    if 0.8 < margin < 1.25:
        return None                      # 太靠近边界，标注长度撑不住这个判定
    return margin < 1.0


def v2c_done(data, seed):
    """**完成判定看日志的 `Finished.`，不看条数。** choice_eng 只有 18 条、
    qa_eng 20、many_shot 54、summary 70、vt 90 —— 按条数过滤会把这些**完整**的
    panel 当成截断的丢掉（本项目已犯过两次）。"""
    f = os.path.join(_LOG, f"v2cbench_{data}_s{seed}.log")
    try:
        return "Finished." in open(f, errors="ignore").read()[-4000:]
    except OSError:
        return False
PANEL = {"scbench_kv": "Retr.KV", "scbench_prefix_suffix": "Retr.PrefSuf",
         "scbench_repoqa": "Code.RepoQA", "squad": "SQuAD", "gsm": "GSM8K",
         "scbench_qa_eng": "En.QA", "scbench_choice_eng": "En.MultiChoice",
         "scbench_summary": "En.Summary", "scbench_vt": "Retr.MultiHop",
         "scbench_mf": "Math.Find", "scbench_many_shot": "ICL.ManyShot"}
# 质心的 tag 是分几批跑出来的，命名不统一，这里显式列出而不是拼规则——
# 拼规则拼错会静默返回空字典，然后整格显示成"缺数据"，很难发现。
# 目录名的形状是 `fastkvzip_<tag>_chunk16k_w4096_cen<K>` —— tag 与 `_cen<K>` 之间
# 还夹着 `_chunk16k_w4096`。首版漏了中间那段，四列全空且不报错（`per_sample` 找不到
# 目录就返回空字典），只能靠"整片都是 —"发现。
_C = "_chunk16k_w4096_cen"
CEN = {
    "scbench_kv": {16: f"__cen16{_C}16", 1024: f"__cen1024{_C}1024"},
    "scbench_vt": {16: f"__p2_cen16_vt{_C}16", 1024: f"__p2_cen1024_vt{_C}1024"},
    "scbench_prefix_suffix": {16: f"__p2_cen16_ps{_C}16",
                              1024: f"__p2_cen1024_ps{_C}1024"},
    "scbench_repoqa": {16: f"__p2_cen16_rq{_C}16", 1024: f"__p2_cen1024_rq{_C}1024"},
}
for d in ("gsm", "squad", "scbench_qa_eng", "scbench_choice_eng",
          "scbench_summary", "scbench_mf", "scbench_many_shot"):
    CEN[d] = {16: f"__cen16_{d}{_C}16", 1024: f"__cen1024_{d}{_C}1024"}
# 另外两批：`_c23*` 覆盖 ratio 0.3/0.2（6 panel），`_r05c*` 覆盖 0.05/0.1（5 panel）。
# 同一 (panel, K) 的不同 ratio 散在不同 tag 里，所以取值时要按 ratio 找对应那批。
CEN23 = {d: {16: f"__c2316_{d}{_C}16", 1024: f"__c231024_{d}{_C}1024"}
         for d in ("scbench_kv", "scbench_mf", "scbench_prefix_suffix",
                   "scbench_repoqa", "scbench_summary", "scbench_vt")}
CEN05 = {d: {16: f"__r05c16{_C}16", 1024: f"__r05c1024{_C}1024"}
         for d in ("scbench_kv", "scbench_prefix_suffix", "scbench_repoqa",
                   "scbench_summary", "scbench_vt")}


def cen_sfx(d, K, r):
    """同一 (panel,K) 的不同 ratio 分散在三批 tag 里，按 ratio 选对应那批。"""
    if r in (0.3, 0.2) and d in CEN23:
        return CEN23[d][K]
    if r == 0.05 and d in CEN05:
        return CEN05[d][K]
    return CEN[d][K] if r == 0.1 else None

_M = contextlib.redirect_stdout(io.StringIO())


@functools.lru_cache(maxsize=None)
def _answers(data):
    """`parse_answer` 会把整个数据集的答案重新解析一遍，而 `per_sample`
    每格都调它一次 —— 全表要调几千次。按数据集缓存。"""
    with _M:
        return parse_answer(data)


@functools.lru_cache(maxsize=None)
def per_sample(data, suffix, ratio):
    """逐样本分数。**已缓存** —— 同一个 (panel, tag, ratio) 在一次生成里会被
    读很多遍：逐格完整性门对每个种子 × 每个 ratio × 每个候选模板各试一次，
    仅这一项就是主循环的几十倍。

    ⚠ 返回的 dict 被缓存**共享**，调用方**不得就地修改**。
    现有调用方只做 `set()` / 索引 / `len()`，都是只读的。
    """
    ANSW, SUBT = _answers(data)
    out, i = {}, 0
    while True:
        f = f"results/{data}/{i}_qwen2.5-7b-instruct-1m_fastkvzip{suffix}/output-pair.json"
        if not os.path.exists(f):
            break
        dd = json.load(open(f))
        p, ans = [], []
        for fmt in [k for k in dd if k.startswith("qa")]:
            for info, txt in dd[fmt]:
                if abs(float(info[0]) - ratio) < 1e-9:
                    p.append(txt["pruned"]); ans.append(txt["answer"])
        gold = ANSW[i] if ANSW else ans
        sub = SUBT[i] if SUBT else None
        if p:
            with _M:
                out[i] = float(np.mean(evaluate_answer(p, gold, data, "qa",
                                                       subtask=sub)))
        i += 1
    return out


def boot(d, n=4000, seed=0):
    r = np.random.default_rng(seed)
    b = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def cell(base, arm):
    c = sorted(set(base) & set(arm))
    if not c:
        return None
    d = (np.array([arm[j] for j in c]) - np.array([base[j] for j in c])) * 100
    m, lo, hi = boot(d)
    return m, (lo > 0 or hi < 0), len(c)


def _seed_ps(d, tpls, S, r, want=None):
    """按顺序试多个 tag 模板，返回第一个非空的逐样本结果。

    一条臂的数据可能分散在不同批次的 tag 里（例如 scalar 的过夜扫描 `_sc11_s*`
    不含 scbench_kv 的 ρ=0.1，那一格只存在于更早的 `_d10scalar_s*`）。
    **按顺序取第一个有数据的**，而不是合并——避免同一 (panel,ratio) 混批。
    """
    # `want` = 该 ratio 下基线的样本数。给了就**优先返回条数吻合的模板**。
    #
    # ⚠ 为什么必须这样：旧版返回「第一个非空」，于是一个**跑到一半**的新 tag
    # 会把一份**跑满**的旧 tag 挡住，逐格完整性检查再把它剔掉 ——
    # **整个种子就这么悄悄消失了**（2026-08-21 03:40 实测：Retr.KV@0.1 的
    # `chd10` 显示 `(2)` 而不是 `(3)`，因为 seed 1 的网格作业才 84/100，
    # 而它 100/100 的旧 tag `_chd10s1` 从没被试到）。
    first = {}
    for t in tpls:
        # 模板可含 `{d}` —— 走 qrun.sh 的全网格必须每 panel 一个 tag，
        # 否则 11 个 panel 会写进同一个 `${TAG#_}.log`、互相覆盖，
        # 完成判定与审计都失效。老模板不含 `{d}`，format 多给一个键无害。
        x = per_sample(d, t.format(S=S, d=d), r)
        if not x:
            continue
        if want is None or len(x) == want:
            return x
        if not first:
            first = x                      # 都不吻合时回退到第一个非空，
    return first                           # 让逐格检查去判它、而不是在这里静默丢


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--write", metavar="RESULTS_GRID.md", default=None,
                    help="把生成的表**写回**该文件，替换首个表头行起的整块表格。"
                         "先前只能靠人工把 stdout 贴进去 —— 在无人值守的循环里，"
                         "一个 `| tail -12` 就把输出截断了而毫无提示。")
    a = ap.parse_args()
    # 多种子臂写成 ("SEEDS", tag 模板, 完成判定)；单 ckpt 臂给 lambda；质心给 None。
    # **`scalar` / `kv` 是原版 v2 代码（`scratch_ctrl_train.py --arch`）训的**，
    # 不是干净版 v2（`varikv_v2.py` 只重写了记忆架构，没有因子臂）。三者共用同一批
    # teacher trace `scratch_ctrl_traces_v2_10`（10 篇、8/2 划分），只是 `--arch` 不同。
    # **覆盖面（2026-08-19 更新）**：`scalar` 现由过夜扫描 `_sc11_s*` 覆盖
    # **11 panel x 7 ratio x 2 种子**；Retr.KV 的 ρ=0.1 在该批里按计划未跑，
    # 回退到旧的 `_d10scalar_s*`（三种子）。`kv` 仍只有 Retr.KV 与 PrefSuf@0.2。
    # 其余格显示 `—` 是**没跑**，不是跑失败。
    ARMS = [("v2", lambda d: "__g8v2_chunk16k_w4096_ctrlmmemo8"),
            ("v2c", ("SEEDS", "__v2c_s{S}_chunk16k_w4096_ctrlmmemo8", v2c_done)),
            # 每条臂可给**多个 tag 模板**，按顺序取第一个有数据的。
            # scalar：过夜扫描 `_sc11_s*` 覆盖 11 panel x 7 ratio x 2 种子；
            # scbench_kv 的 ρ=0.1 在该批里**按计划未跑**（旧批已有三种子），
            # 故回退到 `_d10scalar_s*`。**同一行可能混用两个 tag，见表头说明。**
            ("scalar", ("SEEDS",
                        ("__sc11_s{S}_chunk16k_w4096_ctrlmmemo8_scalar",
                         "__d10scalar_s{S}_chunk16k_w4096_ctrlmstat8_scalar"), None)),
            ("kv", ("SEEDS", ("__d10kv_s{S}_chunk16k_w4096_ctrlmstat8_kv",
                              "__pskv_s{S}_chunk16k_w4096_ctrlmmemo8_kv"), None)),
            ("v3", lambda d: "__g8v3_chunk16k_w4096_ctrlmmemo8"),
            # ---- 2026-08-20 新增三条：全局尺度族 ----
            # `sgs`  = scalar 架构 + `scale=global`（界用 σ_g 而非 σ_h），α=.999
            # `chead`= **严格保序**逐头平移，α=1.25（沿用已撤回的 C* 选 α）
            # `chd10`= 同上但 α=.999，与 `sgs` **α 匹配**，是保序充分性的判决臂
            # 全网格 tag 含 panel（qrun.sh 的日志名由 tag 派生，见 `_seed_ps`）；
            # 第二个模板是更早只跑 ρ=0.1 的那批，作为回退。
            ("sgs", ("SEEDS",
                     ("__gsgs{S}_{d}_chunk16k_w4096_ctrlmmemo8_scalar",
                      "__sgs{S}_chunk16k_w4096_ctrlmmemo8_scalar"), None)),
            ("chead", ("SEEDS",
                       ("__gchd{S}_{d}_chunk16k_w4096_ctrlmmemo8_chead",
                        "__cheads{S}_chunk16k_w4096_ctrlmmemo8_chead"), None)),
            ("chd10", ("SEEDS",
                       ("__gc10{S}_{d}_chunk16k_w4096_ctrlmmemo8_chead",
                        "__chd10s{S}_chunk16k_w4096_ctrlmmemo8_chead"), None)),
            ("cen16", None), ("cen1024", None)]     # 质心按 ratio 选 tag
    agg = {a_: {r: [] for r in RAT} for a_, _ in ARMS}
    rows = []
    for d, name in PANEL.items():
        B = {r: per_sample(d, BASE_SFX.get(r, "__g8base_chunk16k_w4096"), r)
             for r in RAT}
        full = np.mean(list(B[1.0].values())) * 100 if B[1.0] else float("nan")
        # **基线绝对分单独占一行。** 表里其余格全是「与**同 ratio** 基线的配对差」，
        # 只给 `full` 一列时读者无法算出 headroom、也还原不出臂的绝对分 ——
        # 而基线在某些 ratio 上**高过满缓存**（Retr.KV@0.5 = 71.60 > 68.20），
        # 那里 Δ 为负**不等于**掉到满缓存以下。把基线摆出来，这类误读才不可能发生。
        rows.append((name, full, "base",
                     {r: (np.mean(list(B[r].values())) * 100, False, len(B[r]))
                      for r in RAT[1:] if B.get(r)}, d))
        for a_, sfx in ARMS:
            seeds = None
            seed_tpl = None
            try:
                if isinstance(sfx, tuple) and sfx[0] == "SEEDS":
                    _tpl, _done = sfx[1], sfx[2]
                    if isinstance(_tpl, str):
                        _tpl = (_tpl,)
                    if _done is not None:
                        seeds = [S for S in (0, 1, 2) if _done(d, S)]
                    else:
                        # 没有专用完成日志时的通用完备性判定：**要求该臂的样本数
                        # 与本 panel 基线的样本数相等**。比固定阈值稳（choice_eng
                        # 只有 18 条也算完整），也比只看"有没有目录"稳（能挡住
                        # 跑到一半的作业）。
                        # 旧版固定用 ρ=0.2 判完整，于是**只跑了别的 ratio 的臂
                        # 会被整条丢掉**（例如 σ_g 标量最初只跑 ρ=0.1，那一格是
                        # 满的 n=100，却因 0.2 没数据而整行变 `—`）。
                        # 改成：任一 ratio 上与基线样本数相等即入选，
                        # **完整性由下面的逐格检查兜底**（更严，且是按格判的）。
                        seeds = []
                        for S in (0, 1, 2):
                            for _r in RAT:
                                _nb = len(B.get(_r) or {})
                                if _nb and len(_seed_ps(d, _tpl, S, _r, _nb)) == _nb:
                                    seeds.append(S)
                                    break
                    seed_tpl = _tpl
                    A = {}
                elif sfx is None:                        # 质心：逐 ratio 找 tag
                    K = 16 if a_ == "cen16" else 1024
                    A = {}
                    for r in RAT:
                        t = cen_sfx(d, K, r)
                        if t:
                            A[r] = per_sample(d, t, r)
                else:
                    A = {r: per_sample(d, sfx(d), r) for r in RAT}
            except Exception:
                A, seeds = {}, None
            got = {}
            for r in RAT[1:]:
                if not B.get(r):
                    continue
                if seeds is not None:                    # 多种子：逐种子算再平均
                    ms = []
                    for S in seeds:
                        _ps = _seed_ps(d, seed_tpl, S, r, len(B[r]))
                        # **逐格完整性**：该种子在这个 ratio 上的样本数必须与基线
                        # 相等。跑到一半的作业会给出偏斜的交集（本项目已被坑过：
                        # 38/100 时 Math.Find 读到 −3.95★，满量后是 −2.33 不显著）。
                        if len(_ps) != len(B[r]):
                            continue
                        c = cell(B[r], _ps)
                        if c:
                            ms.append(c)
                    if not ms:
                        continue
                    m_ = float(np.mean([x[0] for x in ms]))
                    sig = all(x[1] for x in ms)          # 全部种子都显著才给 ★
                    sd_ = float(np.std([x[0] for x in ms])) if len(ms) > 1 else None
                    got[r] = (m_, sig, len(ms), sd_)
                    agg[a_][r].append(m_)
                    continue
                if not A.get(r):
                    continue
                c = cell(B[r], A[r])
                if c:
                    got[r] = (c[0], c[1], 1)      # 单 ckpt ⇒ 种子数 1
                    agg[a_][r].append(c[0])
            rows.append((name, full, a_, got, d))
    W = 15
    _buf = []
    def print(*a, **k):                    # noqa: A001  仅在本函数内遮蔽
        _buf.append(" ".join(str(x) for x in a))
    hd = f"| {'panel':<15}| {'full':>5} | {'arm':<8}|" + "".join(
        f" {('ρ=%g' % r):>{W}} |" for r in RAT[1:])
    print(hd)
    print("|" + "|".join(["-" * 16, "-" * 7, "-" * 9]
                         + ["-" * (W + 2)] * (len(RAT) - 1)) + "|")
    last = None
    for name, full, a_, got, name_d in rows:
        n = name if name != last else ""
        last = name
        line = f"| {n:<15}| {full:>5.1f} | {a_:<8}|"
        for r in RAT[1:]:
            if r not in got:
                line += f" {'—':>{W}} |"
            else:
                m, sig, ns = got[r][0], got[r][1], got[r][2]
                if a_ == "base":                 # 绝对分，不是 Δ
                    line += f" {m:>{W}.2f} |"
                    continue
                sd = got[r][3] if len(got[r]) > 3 else None
                _d = None if r >= 1.0 else _degenerate(name_d, r)
                if _d is False or r >= 1.0:
                    deg = ""
                elif _d is True:
                    deg = "°"                    # 全格结构性退化
                elif _d is None:
                    deg = "?"                    # 判不了（标注长度撑不住）
                else:
                    deg = "~"                    # **部分**样本退化，占比见脚注
                # **所有臂都标种子数**：v2/v3/质心那几行全是 n=1（表里 v2 用的是
                # 单个 ckpt `ctrl_b_a1_s0`；`+4.27 ± 0.19` 的三种子数字只存在于
                # scbench_kv @0.1 那一格，从没有 11×8 的三种子版本）。只给 v2c 标
                # 会让人误以为别人是多种子的。
                # **只有多种子的格子标 (n) 与散布**：单种子是常态（表头已声明），
                # 每格都印 `(1)` 只是噪声。n≥2 时印 `±跨种子标准差(n)`。
                body = "%+.2f" % m if ns < 2 else "%+.2f±%.2f" % (m, sd or 0.0)
                tail = ("★" if sig else "") + deg + (f"({ns})" if ns >= 2 else "")
                line += f" {body + tail:>{W}} |"
        print(line)
    print("|" + "|".join(["-" * 16, "-" * 7, "-" * 9]
                         + ["-" * (W + 2)] * (len(RAT) - 1)) + "|")
    for a_, _ in ARMS:
        line = f"| {'**均值**':<15}| {'':>5} | {a_:<8}|"
        for r in RAT[1:]:
            v = agg[a_][r]
            line += f" {(('%+.2f (%d)' % (np.mean(v), len(v))) if v else '—'):>{W}} |"
        print(line)

    out = "\n".join(_buf)
    if not a.write:
        import builtins; builtins.print(out); return
    # 本脚本在 import 时 chdir 到 prefill/，所以相对路径必须按**脚本所在目录**解析，
    # 否则会去 prefill/ 下找 RESULTS_GRID.md 并 FileNotFoundError。
    if not os.path.isabs(a.write):
        a.write = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.write)
    doc = open(a.write).read().split("\n")
    # 表格 = 从首个以 "| panel" 开头的行到文件末尾最后一个 "|" 开头的行
    first = next(i for i, L in enumerate(doc) if L.startswith("| panel"))
    last = max(i for i, L in enumerate(doc) if L.startswith("|"))
    open(a.write, "w").write("\n".join(doc[:first] + out.split("\n")
                                       + doc[last + 1:]))
    import builtins
    builtins.print(f"已写回 {a.write}：替换第 {first+1}–{last+1} 行，"
                   f"新表 {len(out.splitlines())} 行")


if __name__ == "__main__":
    raise SystemExit(main())
