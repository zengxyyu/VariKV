"""2×2 因果分解：learned 模块是「记忆」、「steering 向量」，还是两者经 prefill 轨迹的交互？

**为什么必须重做。** 旧的 `scratch_probe_memswap.py` 在 **prefill 完成之后**才换/清零
记忆，而记忆在 prefill 期间已经参与了每个 chunk 的注入 ⇒ 影响 hidden states ⇒ 影响
门控分数 ⇒ 影响**哪些 KV 被驱逐**。所以那个探针只能回答"decode 阶段还需不需要最终
记忆状态"，**不能**回答"记忆内容有没有被用过"。
（类比：让你边读书边做笔记，读完再把笔记换成别人的——你仍答得出来，但这不能证明
你的笔记没帮你理解这本书。）

**正确设计**：把三件事拆成独立开关，四臂各自**从第一个 chunk 起**就配置好。

| 臂 | absorb 内容 | prefill 期读出 | decode 期读出 | 含义 |
|---|---|---|---|---|
| **B** | ❌ | ❌ | ❌ | FastKVzip 基线 |
| **S** | ❌ | ✅ | ✅ | 纯 steering（注入常量，内容全程为空） |
| **M** | ✅ | ❌ | ✅ | 纯记忆（不改写 prefill 轨迹，只在 decode 读出） |
| **F** | ✅ | ✅ | ✅ | 当前完整方法 |

判读（预注册）：

    S ≈ F           ⇒ 它基本就是 steering，不是记忆
    M ≫ B           ⇒ 它确实存了被驱逐的信息
    F ≫ max(S, M)   ⇒ 记忆内容与 prefill 轨迹存在强交互 ⇒ "记忆"故事需要重新理解
    S ≈ B 且 M ≈ B  ⇒ 两条路都没用，增益来自别处

分解：
    steering 效应 = S − B
    记忆效应     = M − B
    交互         = F − S − M + B

实现要点：`read_enabled` 而不是 `residual_mode` 控制"是否注入"。用 residual_mode=False
会让 `_refresh_memory` 把记忆塞进 cache（旧的 KV-injection 路径），那是另一个方法。

用法：CUDA_VISIBLE_DEVICES=k .venv/bin/python scratch_probe_4arm.py --start 0 --n 10
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

_MUTE = contextlib.redirect_stdout(io.StringIO())
ARMS = (("B", False, False, False), ("S", False, True, True),
        ("M", True, False, True), ("F", True, True, True))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="varikv/ckpt_kl/s2b_point_k16.pt")
    ap.add_argument("--gate_scale", type=float, default=0.5)
    ap.add_argument("--data", default="scbench_kv")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
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
    print(f"[cfg] {a.ckpt} mode={ck['mode']} gate_scale={a.gate_scale} "
          f"样本 [{a.start},{a.start + a.n})", flush=True)

    ds = load_dataset_all(a.data, m.tokenizer)
    dw = DataWrapper(a.data, ds, m)
    with _MUTE:
        ANSW, SUBT = parse_answer(a.data)

    def gen_score(i, kv):
        preds = [m.generate(m.apply_template(get_query("qa", q)).to(m.device), kv)
                 for q in list(ds[i]["question"])]
        gold = ANSW[i] if ANSW else list(ds[i]["answers"])
        with _MUTE:
            s = float(np.mean(evaluate_answer(preds, gold, a.data, "qa",
                                              subtask=SUBT[i] if SUBT else None)))
        return s, preds

    rows, same_SF = [], 0
    for i in range(a.start, a.start + a.n):
        sc, pr = {}, {}
        for nm, absorb, pre_read, dec_read in ARMS:
            if nm == "B":                       # 完全无记忆：换 kv_type
                _kt = m.kv_type; m.kv_type = "retain"
                kv = dw.prefill_context(i, prefill_chunk=16000, window_size=4096,
                                        chunk_ratio=a.ratio, level="pair")
                m.kv_type = _kt
            else:
                m.varikv_absorb_content = absorb
                m.varikv_read_enabled = pre_read     # **从第一个 chunk 起**就生效
                kv = dw.prefill_context(i, prefill_chunk=16000, window_size=4096,
                                        chunk_ratio=a.ratio, level="pair")
                kv.read_enabled = dec_read           # decode 阶段单独设
            sc[nm], pr[nm] = gen_score(i, kv)
            del kv; torch.cuda.empty_cache()
        same_SF += int(pr["S"] == pr["F"])
        rows.append([sc[n] * 100 for n, *_ in ARMS])
        print(f"  样本{i}: B {sc['B']*100:5.1f}  S {sc['S']*100:5.1f}  "
              f"M {sc['M']*100:5.1f}  F {sc['F']*100:5.1f}", flush=True)

    A = np.array(rows)
    B, S, M, F = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    print("\n" + "=" * 76)
    print(f"2×2 因果分解　{len(A)} 条　{a.data} @ratio {a.ratio}　gate_scale {a.gate_scale}")
    print("-" * 76)
    for nm, lbl, v in (("B", "基线（无记忆）", B), ("S", "纯 steering（内容全程为空）", S),
                       ("M", "纯记忆（prefill 不注入）", M), ("F", "完整方法", F)):
        print(f"  {nm}  {lbl:<28}{v.mean():>7.2f}")
    print("-" * 76)
    print(f"  steering 效应  S−B = {S.mean()-B.mean():+7.2f}")
    print(f"  记忆效应       M−B = {M.mean()-B.mean():+7.2f}")
    print(f"  完整增益       F−B = {F.mean()-B.mean():+7.2f}")
    print(f"  **交互**  F−S−M+B = {F.mean()-S.mean()-M.mean()+B.mean():+7.2f}")
    print(f"\n  S 与 F 预测串逐字相同：{same_SF}/{len(A)} ({100*same_SF/len(A):.0f}%)")
    print("=" * 76)
    print("判读：S≈F ⇒ 是 steering；M≫B ⇒ 确实存了信息；F≫max(S,M) ⇒ 强交互")


if __name__ == "__main__":
    main()
