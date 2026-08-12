"""自动生成 ckpt 清单 —— 直接读 .pt 的真实内容，不靠记忆。

用法：
    .venv/bin/python scratch_model_registry.py            # 打印表格
    .venv/bin/python scratch_model_registry.py --md       # markdown（贴进 MODELS.md）

为什么要自动生成：ckpt 有 40+ 个、分散在 11 个目录，手写清单必然过期。
旧 ckpt 只存了 memory/mode/num_slots/model 四个键，训练配置无法从 .pt 复原
（2026-08-12 起 argparse namespace 随 ckpt 一起存，见 CLAUDE.md）；本脚本对
缺 args 的 ckpt 输出 "?"，正好把"哪些 ckpt 的配置不可复原"显式标出来。
"""
import argparse
import datetime as dt
import glob
import os

import torch

# 每代 ckpt 的用途与关键特征（唯一需要人工维护的部分；键 = 目录名）
GENERATION = {
    "ckpt": ("stage1", "1.5B 合成 needle 任务，K∈{2,4,8,16,32,64} × tier{2,4,5}"),
    "ckpt_real": ("stage2a", "1.5B 真实语料 fineweb-edu 长文档 LM"),
    "ckpt_stage2b": ("stage2b-v0", "首次接入 harness；训练配置 2048/256/8k ≠ 评测 16000/4096"),
    "ckpt_stage2b_matched": ("stage2b-v1", "训练配置对齐评测"),
    "ckpt_stage2b_retain": ("stage2b-v2", "改建在 RetainCache 上（基线所用机制）"),
    "ckpt_stage2b_res": ("stage2b-res", "残差读出 + --obj lm（门单调打开，下游崩）"),
    "ckpt_gap_fix03": ("stage2b-gap", "残差 + --obj gap，固定 ratio 0.3"),
    "ckpt_gap_rand": ("stage2b-gap", "残差 + --obj gap，每步随机 ratio"),
    "ckpt_kl": ("v1-KL", "**残差 + 多位置 teacher KL**（首个下游正结果）"),
    "ckpt_kl_v2a": ("v2a-KL", "v1 配置 + 全部修复，但 min_chunks=1 误滤成 14/34 篇"),
    "ckpt_kl_v2s": ("v2s-KL", "流式：10 篇长文档、每步 4 次驱逐、800 步"),
    "ckpt_kl_v2b": ("v2b-KL", "v1 的干净复现：修复后代码 + min_chunks 0（全 34 篇）"),
}

# 关键训练参数（有则列出；缺 = 该 ckpt 训练时还没有 argparse 落盘）
KEYS = ("obj", "mode", "num_slots", "ratio", "ratio_mode", "max_ctx", "chunk",
        "window", "target_len", "steps", "lr", "gate_lr", "residual",
        "ctx_pos", "kl_weight", "min_chunks", "detach_every", "n_short",
        "n_long", "seed")


def scan():
    rows = []
    for f in sorted(glob.glob("varikv/ckpt*/**/*.pt", recursive=True)):
        d = os.path.basename(os.path.dirname(f))
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:                                   # noqa: BLE001
            rows.append({"path": f, "err": str(e)[:40]})
            continue
        sd = ck.get("memory", {})
        a = ck.get("args") or {}
        gate = sd.get("residual_gate")
        g = torch.sigmoid(gate.float()) if gate is not None else None
        rows.append({
            "path": f, "dir": d,
            "gen": GENERATION.get(d, ("?", "?"))[0],
            "note": GENERATION.get(d, ("?", "?"))[1],
            "mtime": dt.datetime.fromtimestamp(os.path.getmtime(f)),
            "mode": ck.get("mode", "?"), "K": ck.get("num_slots", "?"),
            "model": ck.get("model", "?"),
            "params": sum(v.numel() for v in sd.values() if hasattr(v, "numel")),
            "has_args": bool(a),
            "args": {k: a[k] for k in KEYS if k in a},
            "gate_mean": float(g.mean()) if g is not None else None,
            "gate_max": float(g.max()) if g is not None else None,
            "gate_open": float((g > 0.1).float().mean()) if g is not None else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    rows = [r for r in scan() if "err" not in r]
    short = lambda m: (m or "?").split("/")[-1].replace("-Instruct", "")   # noqa: E731

    if args.md:
        print("| ckpt | 代 | 训练完成 | 模型 | mode | K | 参数 | σ(gate) mean/max | args 可复原 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            gs = (f"{r['gate_mean']:.3f} / {r['gate_max']:.3f}"
                  if r["gate_mean"] is not None else "—（无残差门）")
            print(f"| `{r['path']}` | {r['gen']} | {r['mtime']:%m-%d %H:%M} | "
                  f"{short(r['model'])} | {r['mode']} | {r['K']} | "
                  f"{r['params']/1e6:.2f}M | {gs} | "
                  f"{'✅' if r['has_args'] else '❌ 只有 4 个键'} |")
        print("\n### 有完整训练配置的 ckpt\n")
        for r in rows:
            if r["has_args"]:
                kv = "  ".join(f"{k}={v}" for k, v in r["args"].items())
                print(f"- **`{r['path']}`**\n  `{kv}`")
        return

    print(f"{'ckpt':<44}{'代':<12}{'完成':<13}{'mode':<7}{'K':>4}"
          f"{'门 mean/max':>16}{'args':>6}")
    print("-" * 104)
    for r in rows:
        gs = (f"{r['gate_mean']:.3f}/{r['gate_max']:.3f}"
              if r["gate_mean"] is not None else "—")
        print(f"{r['path']:<44}{r['gen']:<12}{r['mtime']:%m-%d %H:%M}  "
              f"{r['mode']:<7}{r['K']:>4}{gs:>16}{'Y' if r['has_args'] else 'n':>6}")
    print("-" * 104)
    n_args = sum(1 for r in rows if r["has_args"])
    print(f"共 {len(rows)} 个 ckpt，其中 {n_args} 个可从 .pt 复原训练配置，"
          f"{len(rows)-n_args} 个只能靠日志反推")


if __name__ == "__main__":
    main()
