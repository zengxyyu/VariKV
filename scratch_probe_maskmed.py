"""mask 变化是性能变化的因果中介吗？—— 第二个 2×2（selection × representation）

**背景。** 第一个 2×2（`scratch_probe_4arm.py`，80 条）测出全部 +40 来自
「吸收内容 × 预填期注入」的交互。随后 `scratch_probe_evictshift.py` 测到保留集合确实
变了。但那只给出

    feedback ⇒ mask 变化      和      feedback ⇒ 性能变化

**推不出** `mask 变化 ⇒ 性能变化`。预填注入同时改变了 `h/q/k/v` 和后续所有层，
所以即使保留的 token id 完全相同，**存下来的 K/V 数值也不同**。

**设计。** 冻结 mask 的反事实，构成第二个 2×2：

|                | B 的 mask | F 的 mask |
|---|---|---|
| **B 的表示**（无记忆） | `B` | **`B|R_F`** |
| **F 的表示**（有反馈） | **`F|R_B`** | `F` |

    selection 效应      = B|R_F − B
    representation 效应 = F|R_B − B
    交互                = F − F|R_B − B|R_F + B

判读（预注册）：
    F|R_B ≈ B 且 B|R_F ≫ B  ⇒ **收益由 selection 中介**（学到了更好的驱逐策略）
    F|R_B ≈ F 且 B|R_F ≈ B  ⇒ **收益由 representation 中介**（mask 变化只是副产品）
    两者都居中               ⇒ 两条通路都有贡献（我目前认为最可能）

**冻结实现**：`RetainCache.prune_chunk` 只做「算出本段 valid 再拼接」，所以逐 chunk
录下新增的那一段、回放时覆盖即可。覆盖必须**重新绑定**而不是原地写 —— 预填在
`inference_mode` 下，原地改 inference tensor 会直接报错。
注意 `MemoryRetainCache.prune_chunk` 先调 `super().prune_chunk()` 再按 `self.valid`
吸收，所以回放后**吸收用的也是被强制的那个驱逐集合**，这正是我们想要的
「压缩完全相同，只有反馈不同」。

**同时修正 mask 统计**：旧探针把逐 (层,头) 的比例先算再平均，而 `level="pair"`
全局分配预算 ⇒ 逐对保留量从 4096 到 41354 差 10 倍，比例平均被严重扭曲
（旧报「B 丢 1.76% / F 新增 9.83%」自相矛盾：全局 |B|=|F| 时两者必须相等）。
本脚本一律**先全局求和再相除**。

用法：CUDA_VISIBLE_DEVICES=k .venv/bin/python scratch_probe_maskmed.py --start 0 --n 5
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip                     # noqa: E402
from attention.kvcache import RetainCache                # noqa: E402
from data.load import load_dataset_all                   # noqa: E402
from data.wrapper import DataWrapper, get_query          # noqa: E402
from results.parse import parse_answer, evaluate_answer  # noqa: E402

_MUTE = contextlib.redirect_stdout(io.StringIO())
ST = {"mode": "off", "rec": [], "replay": None, "k": 0}
_orig_prune = RetainCache.prune_chunk


def _patched(self, ratio, evict_range=tuple, level="pair"):
    out = _orig_prune(self, ratio, evict_range, level)
    n_new = evict_range[1] - evict_range[0]
    if ST["mode"] == "rec":
        ST["rec"].append(self.valid[..., -n_new:].detach().clone().cpu())
    elif ST["mode"] == "replay":
        f = ST["replay"][ST["k"]].to(self.valid.device)
        assert f.shape[-1] == n_new, f"chunk 长度不符 {f.shape[-1]} vs {n_new}"
        # **重新绑定**而不是原地写：预填在 inference_mode 下，原地改会报错
        self.valid = torch.cat([self.valid[..., :-n_new], f], dim=-1)
        ST["k"] += 1
    return out


RetainCache.prune_chunk = _patched


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="varikv/ckpt_kl/s2b_point_k16.pt")
    ap.add_argument("--gate_scale", type=float, default=0.5)
    ap.add_argument("--data", default="scbench_kv")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--ratio", type=float, default=0.1)
    a = ap.parse_args()

    m = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M", kv_type="memory_retain",
                   gate_path_or_name="fastkvzip")
    from varikv.config import Config
    from varikv.memory import DistributionalMemory
    ck = torch.load(ROOT / a.ckpt, map_location=m.device)
    cfg = Config(); cfg.memory.num_slots = ck["num_slots"]
    H = m.config.num_key_value_heads; L = m.config.num_hidden_layers
    d = getattr(m.config, "head_dim", m.config.hidden_size // m.config.num_attention_heads)
    mem = DistributionalMemory(2 * d, cfg.memory, mode=ck["mode"],
                               n_groups=L * H).to(m.device, dtype=torch.float32)
    mem.load_state_dict(ck["memory"]); mem.eval()
    m.varikv_memory = mem; m.varikv_M = ck["num_slots"]; m.varikv_residual = True
    m.varikv_gate_scale = a.gate_scale
    rot = getattr(m.model.model, "rotary_emb", None)
    m.varikv_inv_freq = rot.inv_freq.detach().clone()
    print(f"[cfg] {a.ckpt} gate×{a.gate_scale} 样本 [{a.start},{a.start+a.n})", flush=True)

    ds = load_dataset_all(a.data, m.tokenizer)
    dw = DataWrapper(a.data, ds, m)
    with _MUTE:
        ANSW, SUBT = parse_answer(a.data)
    P = dict(prefill_chunk=16000, window_size=4096, chunk_ratio=a.ratio, level="pair")

    def sc(i, kv):
        preds = [m.generate(m.apply_template(get_query("qa", q)).to(m.device), kv)
                 for q in list(ds[i]["question"])]
        gold = ANSW[i] if ANSW else list(ds[i]["answers"])
        with _MUTE:
            return float(np.mean(evaluate_answer(preds, gold, a.data, "qa",
                                                 subtask=SUBT[i] if SUBT else None)))

    def run(i, memory, mode, replay=None):
        ST["mode"] = mode; ST["rec"] = []; ST["replay"] = replay; ST["k"] = 0
        _kt = m.kv_type
        m.kv_type = "memory_retain" if memory else "retain"
        kv = dw.prefill_context(i, **P)
        m.kv_type = _kt
        v = kv.valid.detach().clone().cpu()
        s = sc(i, kv)
        rec = list(ST["rec"]); ST["mode"] = "off"
        del kv; torch.cuda.empty_cache()
        return s, v, rec

    rows = []
    for i in range(a.start, a.start + a.n):
        sB, vB, recB = run(i, False, "rec")           # B：记录 mask
        sF, vF, recF = run(i, True, "rec")            # F：记录 mask
        sFB, _, _ = run(i, True, "replay", recB)      # F|R_B：有反馈，强制 B 的 mask
        sBF, _, _ = run(i, False, "replay", recF)     # B|R_F：无记忆，强制 F 的 mask
        n = min(vB.shape[-1], vF.shape[-1])
        b, f = vB[..., :n].bool(), vF[..., :n].bool()
        # **全局求和再相除**，不做逐对比例平均
        I = int((b & f).sum()); U = int((b | f).sum())
        nb, nf = int(b.sum()), int(f.sum())
        rows.append((sB*100, sBF*100, sFB*100, sF*100, I/U,
                     int((b & ~f).sum())/nb, int((~b & f).sum())/nf, nb, nf))
        print(f"  样本{i}: B {sB*100:5.1f} | B|R_F {sBF*100:5.1f} | "
              f"F|R_B {sFB*100:5.1f} | F {sF*100:5.1f}   "
              f"IoU {I/U:.4f} 换掉 {100*int((b&~f).sum())/nb:.2f}%", flush=True)

    A = np.array(rows)
    B_, BF, FB, F_ = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    def bt(dv, n=10000, seed=0):
        r = np.random.default_rng(seed); ix = r.integers(0, len(dv), (n, len(dv)))
        s = dv[ix].mean(1); return dv.mean(), float(np.quantile(s, .025)), float(np.quantile(s, .975))
    print("\n" + "=" * 84)
    print(f"selection × representation 分解　{len(A)} 条　{a.data} @ratio {a.ratio}")
    print("-" * 84)
    print(f"{'':<22}{'B 的 mask':>14}{'F 的 mask':>14}")
    print(f"{'B 的表示（无记忆）':<22}{B_.mean():>14.2f}{BF.mean():>14.2f}")
    print(f"{'F 的表示（有反馈）':<22}{FB.mean():>14.2f}{F_.mean():>14.2f}")
    print("-" * 84)
    for lbl, dv in (("selection 效应      B|R_F − B", BF - B_),
                    ("representation 效应 F|R_B − B", FB - B_),
                    ("完整增益            F − B", F_ - B_),
                    ("交互 F−F|R_B−B|R_F+B", F_ - FB - BF + B_)):
        mm, lo, hi = bt(dv)
        print(f"  {lbl:<32}{mm:+7.2f} [{lo:+6.2f},{hi:+6.2f}]"
              f"{'★' if (lo > 0 or hi < 0) else ' 未分离'}")
    print("-" * 84)
    print(f"  mask（全局求和口径）IoU {A[:, 4].mean():.4f}　"
          f"B 丢 {100*A[:, 5].mean():.2f}%　F 新增 {100*A[:, 6].mean():.2f}%　"
          f"|B| {A[:, 7].mean():.0f} / |F| {A[:, 8].mean():.0f}")
    print("=" * 84)
    print("判读：F|R_B≈B 且 B|R_F≫B ⇒ selection 中介；F|R_B≈F 且 B|R_F≈B ⇒ representation 中介")


if __name__ == "__main__":
    main()
