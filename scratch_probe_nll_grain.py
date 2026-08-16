#!/usr/bin/env python3
"""效用在**什么粒度**上才是可辨识的 —— 由稳定性对照的失败直接推出的下一问。

`scratch_probe_nll_stab.py` 测到：单条 KV 的 `U^NLL` 在两个只差 1% 的存活集合上
**互不相关**（合并 ρ=−0.22，`std(U_S−U_S')=6.44e-3` 反而大于
`√(var_S+var_S')=6.22e-3`，共享成分 ≤ 0）。所以"这一条 KV 的效用"在阈值附近
**不是一个稳定的量**，任何逐 token 的边际效用标签都被自身的不可复现性淹没。

但压缩显然是有效应的 —— Retr.KV 满缓存 68.20、ratio 0.1 掉到 45 附近。效应真实存在，
只是不驻留在单条上。**那它驻留在什么尺度上？** 这个尺度就是新教师该用的粒度。

做法：从存活集合里按分数 **top / bottom / random** 各去掉 `G` 条，`G` 跨五个数量级
扫，看 `ΔNLL(top)` 与 `ΔNLL(bot)` 从哪个 `G` 起分得开（相对于同 `G` 下 random 的
重抽样散布）。`random` 抽 `--n_rand` 次，给出该粒度下的**噪声带**，这是判据的分母。

    分得开的最小 G  = 效用可辨识的粒度
      G ~ 1–10      ⇒ 逐 token 教师本来就该行，稳定性对照的结论要重查
      G ~ 10³–10⁴   ⇒ 只有成组才有信号 ⇒ 教师应改为**组级/预算级**，
                      与"+4.27 来自跨层头预算再校准"的假设一致
      永远分不开     ⇒ 门控分数与答案 NLL 无关（而不是"靶子错位"），
                      那 −9.96 与 +4.40 都得另找机制

**上一版对照 B 的教训写在这里**：它取 `G=256`，而 `level="pair"` 的保留集是
28 层 × 4 头 × ~165k × 10% ≈ **1.85M 格**，256 只占 **0.0139%**。三臂当然都测不出
差别。挑 `G` 一定要按保留集**总量**的比例来，不要按"每头多少条"的直觉。
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
_P = os.path.join(ROOT, "external/FastKVzip/prefill")
sys.path.insert(0, _P)
os.chdir(_P)

from data import DataWrapper, load_dataset_all                  # noqa: E402
from model import ModelKVzip                                    # noqa: E402
from utils import set_gen_length                                # noqa: E402


@torch.inference_mode()
def nll(model, ids, n_ans, kv):
    p = model._prob(ids, kv)[-n_ans - 1:-1].float()
    lab = ids[0, -n_ans:]
    return float(-p.gather(1, lab[:, None]).clamp_min(1e-12).log().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="scbench_kv")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--fracs", type=float, nargs="+",
                    default=[1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 2e-1],
                    help="去掉保留集的多大比例。按**总量**取，不是每头条数")
    ap.add_argument("--n_rand", type=int, default=4, help="每个 G 的 random 重抽次数")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--chunk", type=int, default=16000)
    ap.add_argument("--window", type=int, default=4096)
    a = ap.parse_args()

    m = ModelKVzip(a.model, "retain", "fastkvzip")
    ds = DataWrapper(a.data, load_dataset_all(a.data, m.tokenizer), m)
    set_gen_length(a.data, m)
    g = torch.Generator(device="cpu").manual_seed(0)
    rows = []                       # (si, frac, arm, rep, dnll)

    for si in range(min(a.num, len(ds))):
        kv_f = ds.prefill_context(si, do_score=False)
        inputs, _ = ds.generate_answer(si, kv_f, prob=False)
        task = "qa" if "qa" in inputs else list(inputs.keys())[0]
        ids = torch.cat([inputs[task][k] for k in ("q", "a")], dim=-1)
        n_ans = len(inputs[task]["a"][0])
        del kv_f
        torch.cuda.empty_cache()

        kv = ds.prefill_context(si, prefill_chunk=a.chunk, window_size=a.window,
                                chunk_ratio=a.ratio, level="pair")
        kv.valid = kv.valid.clone()
        V = kv.valid
        base = nll(m, ids, n_ans, kv)
        sc = torch.stack(kv.score, 0)[:, 0]
        s_flat = sc[..., kv.sink:kv.sink + V.shape[-1]].float().reshape(-1)
        ki = V.reshape(-1).nonzero(as_tuple=True)[0]
        order = s_flat[ki].argsort(descending=True)
        nk = len(ki)
        print(f"  样本 {si}: base {base:.4f}  保留集 {nk/1e6:.2f}M 格", flush=True)

        for fr in a.fracs:
            G = max(int(nk * fr), 1)
            if G > nk // 2:
                continue
            for arm in ("top", "bot", "rand"):
                reps = a.n_rand if arm == "rand" else 1
                for rp in range(reps):
                    if arm == "top":
                        idx = ki[order[:G]]
                    elif arm == "bot":
                        idx = ki[order[-G:]]
                    else:
                        idx = ki[torch.randperm(nk, generator=g)[:G]]
                    V.view(-1)[idx] = False
                    d = nll(m, ids, n_ans, kv) - base
                    V.view(-1)[idx] = True
                    rows.append((si, fr, arm, rp, d))
            r = [x[4] for x in rows if x[0] == si and x[1] == fr]
            print(f"    G={G:>7} ({fr:g})  top {r[0]:+.4f}  bot {r[1]:+.4f}  "
                  f"rand {np.mean(r[2:]):+.4f}±{np.std(r[2:]):.4f}", flush=True)
        del kv
        torch.cuda.empty_cache()

    np.save(os.path.join(ROOT, f"scratch_nllgrain_{a.data}.npy"),
            np.array([(r[0], r[1], {"top": 0, "bot": 1, "rand": 2}[r[2]], r[3], r[4])
                      for r in rows]))
    print(f"\n=== {a.data} @ ratio {a.ratio}　{min(a.num, len(ds))} 篇 ===")
    print(f"{'去掉比例':>10}{'top':>12}{'bot':>12}{'rand 均±散':>20}"
          f"{'(top−bot)/σ_rand':>20}")
    for fr in a.fracs:
        t = np.array([r[4] for r in rows if r[1] == fr and r[2] == "top"])
        b = np.array([r[4] for r in rows if r[1] == fr and r[2] == "bot"])
        rd = np.array([r[4] for r in rows if r[1] == fr and r[2] == "rand"])
        if not len(t):
            continue
        # 分母用 random 在**同一粒度**下的散布：它就是"随便删 G 条"的噪声带
        sd = rd.std() if rd.std() > 1e-12 else float("nan")
        print(f"{fr:>10g}{t.mean():>12.4f}{b.mean():>12.4f}"
              f"{rd.mean():>13.4f}±{rd.std():.4f}{(t.mean()-b.mean())/sd:>20.2f}")
    print("\n判读：|top−bot|/σ_rand 首次稳定超过 ~2 的那个粒度，就是效用可辨识的尺度；"
          "\n      若只在 10⁻²–10⁻¹ 才超过 ⇒ 教师必须组级/预算级，逐 token 标签注定被噪声淹没。")


if __name__ == "__main__":
    raise SystemExit(main())
