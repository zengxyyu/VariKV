"""memory-swap 对照：那 0.33M 模块是「记忆」还是「通用 steering 向量」？

**为什么这是必答题。** `point×0.5` 在 Retr.KV 上拿 64.20（HRR 88.8%）。但取证已经测出
它注入的向量与真正的局部缺口 `Δo` 正交（cos ≈ 0），方向与被驱逐内容无关。那么它
到底是在「把这一段上下文里被删掉的东西找回来」，还是只学会了一个**与上下文无关**的
通用修正向量？审稿人必问，而两种答案指向完全不同的论文。

**设计。** 记忆状态就在 `mem` 的五个张量里（`mu/logvar/var_content/pos/_pos_tau`，
形状 `[1, L*H, K, ·]`），所以可以在预填之后直接覆盖：

    1. 预填样本 j → 记忆吸收 j 的被驱逐 KV → 快照 S_j
    2. 预填样本 i → 记忆吸收 i 的（正常路径）→ 记 score_normal
    3. 把 i 的记忆状态**覆盖成 S_j**（保留缓存仍是 i 的）→ 记 score_swap
    4. 把记忆状态清零（reset，但仍注入）→ 记 score_empty

**判读（预注册）：**
    score_swap ≈ score_normal          ⇒ 与上下文无关 ⇒ **是 steering 向量，不是记忆**
                                          「吸收被驱逐 KV」这个叙事崩塌
    score_swap ≈ baseline 或更差        ⇒ 内容确实是**当前上下文**特有的 ⇒ 是记忆
    score_empty ≈ baseline             ⇒ 注入本身无害，收益来自内容（好）
    score_empty ≫ baseline             ⇒ 连空记忆都有用 ⇒ 收益来自注入这个动作本身

**已知混淆项**：槽带的 `pos`（位置质心）来自 j，而读出要按它做 RoPE 旋转，所以
score_swap 掉分可能部分来自**位置错配**而非内容错配。两个上下文都是 ~169k，帧大致
可比；`--keep_pos` 提供"只换内容、保留 i 的 pos"的变体来隔离这一项。

用法：
    CUDA_VISIBLE_DEVICES=k .venv/bin/python scratch_probe_memswap.py \
        --ckpt varikv/ckpt_kl/s2b_point_k16.pt --gate_scale 0.5 --n 8
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
from data.load import load_dataset_all                   # noqa: E402
from data.wrapper import DataWrapper, get_query          # noqa: E402
from results.parse import parse_answer, evaluate_answer  # noqa: E402

DBG = []
STATE = ("mu", "logvar", "var_content", "pos", "_pos_tau")
_MUTE = contextlib.redirect_stdout(io.StringIO())


def snap(mem):
    return {n: getattr(mem, n).detach().clone() for n in STATE}


def restore(mem, s, keep_pos=False):
    """**重新绑定**而不是原地 copy_。`prefill` 走 `@torch.inference_mode()`，
    它产出的张量是 inference tensor，在 inference mode 之外原地修改会直接报
    "Inplace update to inference tensor outside InferenceMode is not allowed"。
    赋值只换引用、不写内存，所以安全；后续前向读它们没有问题。"""
    for n in STATE:
        if keep_pos and n in ("pos", "_pos_tau"):
            continue
        setattr(mem, n, s[n])


def zero_state(mem):
    for n in STATE:
        setattr(mem, n, torch.zeros_like(getattr(mem, n)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="varikv/ckpt_kl/s2b_point_k16.pt")
    ap.add_argument("--gate_scale", type=float, default=0.5)
    ap.add_argument("--data", default="scbench_kv")
    ap.add_argument("--n", type=int, default=8)   # 必须 ≥2，否则供体退化成自己
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--keep_pos", action="store_true")
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
    print(f"[cfg] ckpt={a.ckpt} mode={ck['mode']} gate_scale={a.gate_scale} "
          f"keep_pos={a.keep_pos}")

    ds = load_dataset_all(a.data, m.tokenizer)
    dw = DataWrapper(a.data, ds, m)
    with _MUTE:
        ANSW, SUBT = parse_answer(a.data)

    def score(i, kv):
        """在样本 i 的问题上生成并打分。

        `m.generate` 返回字符串，且默认 `update_cache=False` ⇒ 生成后自己
        `kv.slice(seen_token_prev)` 回滚，所以这里**不要**再手动 slice。
        字段名是 `answers`（复数），不是 `answer`。
        """
        preds = list()
        for q in list(ds[i]["question"]):
            ids = m.apply_template(get_query("qa", q)).to(m.device)
            preds.append(m.generate(ids, kv))
        if DBG:
            print(f"      [{DBG[0]}] {preds[0][:52]!r}", flush=True)
        gold = ANSW[i] if ANSW else list(ds[i]["answers"])
        with _MUTE:
            return float(np.mean(evaluate_answer(preds, gold, a.data, "qa",
                                                 subtask=SUBT[i] if SUBT else None)))

    rows = []
    for i in range(a.n):
        j = (i + 1) % a.n                   # 配对：拿**另一条**样本的记忆
        assert j != i, "供体必须与受试不同（n=1 时旧逻辑会退化成自己）"
        # ① 供体：预填 j，快照它的记忆
        kv_j = dw.prefill_context(j, prefill_chunk=16000, window_size=4096,
                                  chunk_ratio=a.ratio, level="pair")
        S_j = snap(mem)
        del kv_j; torch.cuda.empty_cache()
        # ② 正常：预填 i
        kv = dw.prefill_context(i, prefill_chunk=16000, window_size=4096,
                                chunk_ratio=a.ratio, level="pair")
        DBG[:] = ["正常"]
        s_norm = score(i, kv)
        S_i = snap(mem)
        # ③ 换成 j 的记忆（缓存仍是 i 的）
        restore(mem, S_j, keep_pos=a.keep_pos)
        DBG[:] = ["换记忆"]
        s_swap = score(i, kv)
        # ④ 空记忆（清零但仍注入）
        zero_state(mem)
        DBG[:] = ["空记忆"]
        s_empty = score(i, kv)
        del kv; torch.cuda.empty_cache()
        # ⑤ **真基线：必须重新预填一次、全程无记忆。**
        # 不能只在生成时关 residual_mode —— 预填期间 attn.py 每个 chunk 都调过
        # memory_residual，记忆因此已影响 hidden states → 门控分数 → 哪些 KV 被驱逐。
        # 那样得到的是"记忆影响过的预填 + 生成时不读出"，实测比真基线高很多
        # （样本 0：假基线 100.0，而 harness 的真基线是 60.0）。
        _kt = m.kv_type
        m.kv_type = "retain"
        kv_b = dw.prefill_context(i, prefill_chunk=16000, window_size=4096,
                                  chunk_ratio=a.ratio, level="pair")
        m.kv_type = _kt
        DBG[:] = ["真基线"]
        s_base = score(i, kv_b)
        del kv_b; torch.cuda.empty_cache()
        restore(mem, S_i)
        rows.append((i, j, s_base * 100, s_norm * 100, s_swap * 100, s_empty * 100))
        print(f"  样本{i}(供体{j}): 基线 {s_base*100:5.1f}  正常 {s_norm*100:5.1f}  "
              f"换记忆 {s_swap*100:5.1f}  空记忆 {s_empty*100:5.1f}", flush=True)

    A = np.array(rows)
    print("\n" + "=" * 78)
    print(f"memory-swap 对照　{len(A)} 条　{a.data} @ratio {a.ratio}")
    print("-" * 78)
    for k, nm in ((2, "基线（不注入）"), (3, "正常（自己的记忆）"),
                  (4, "**换成别的样本的记忆**"), (5, "空记忆（清零仍注入）")):
        print(f"  {nm:<26}{A[:, k].mean():>7.2f}")
    print("-" * 78)
    gain = A[:, 3].mean() - A[:, 2].mean()
    keep = A[:, 4].mean() - A[:, 2].mean()
    print(f"  正常带来的增益          {gain:+7.2f}")
    print(f"  换记忆后剩下的增益      {keep:+7.2f}"
          f"  ({100*keep/gain if abs(gain)>1e-9 else float('nan'):.0f}% 被保留)")
    print("=" * 78)
    print("判读：保留比例 ≈0% ⇒ 内容是上下文特有的（是记忆）；"
          "≈100% ⇒ 与上下文无关（是 steering 向量）")


if __name__ == "__main__":
    main()
