# ==============================================================================
# Benchmark evaluation with chunked-prefill-evict (MRCR, context length <= 128K)
# ==============================================================================

from collections import defaultdict
from difflib import SequenceMatcher


def grade(response: str, answer: str, random_string_to_prepend: str) -> float:
    """Compare response and answer using SequenceMatcher ratio"""
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def set_ratios():
    """Set compression ratios for evaluation"""
    return [1.0, 0.75, 0.5, 0.4, 0.3, 0.2]


if __name__ == "__main__":
    from args import args
    from model import ModelKVzip

    from data import load_dataset_all
    from utils import TimeStamp, save_result

    args.data = "mrcr"
    args.tag += f"_chunk{args.prefill_chunk//1000}k_w{args.window_size}"
    print(f"tag: {args.tag}")

    # 本地改动：MRCR 是 Figure 11 的第 12 个数据集，但本脚本原来不认 --ctrlm_ckpt，
    # 学习残差臂在 MRCR 上无法评测（12 个 panel 只能报 11 个）。这里补上，
    # 与 eval_chunk.py 的 control_learned 分支同一套构造逻辑（typed 从权重形状推断）。
    if getattr(args, "ctrlm_ckpt", ""):
        import torch as _torch
        from attention.control_memory import ControlMemory as _CM
        args.kv_type = "control_learned"
        _ck = _torch.load(args.ctrlm_ckpt, map_location="cpu")
        _mode = args.ctrlm_mode or _ck.get("mode", "stateful")
        _ns = _ck.get("slots", 8)
        args.tag += f"_ctrlm{_mode[:4]}{_ns}"
        model = ModelKVzip(args.model, args.kv_type, args.gate_path_or_name)
        _cm = _CM(_ck.get("d_kv", 128), _ck["L"], _ck["H"], n_slots=_ns,
                  d_m=_ck.get("dim", 128), mode=_mode,
                  typed=_ck["state"]["M_init"].shape[2] == 2 * _ns)
        _cm.load_state_dict(_ck["state"])
        model.ctrl_module = _cm.to(model.device).eval()
        model.ctrl_seed = getattr(args, "ctrl_seed", 0)
        print(f"[CtrlM-MRCR] {args.ctrlm_ckpt} mode={_mode} "
              f"alpha={float(_cm.alpha):.4f}")
    else:
        model = ModelKVzip(args.model, args.kv_type, args.gate_path_or_name)
    dataset = load_dataset_all(args.data, model.tokenizer, n_data=2400)

    tt = TimeStamp(True)
    # 本地改动：原版跑 range(args.idx, len(dataset))，即全部 2400 条 × 6 个压缩比，
    # 在 128K 上下文下约 300 GPU-hours。args.py 早已定义 --num（默认 100），
    # 其余 eval 脚本都用了，只有本脚本漏用；这里补齐，与 eval.py/eval_chunk.py 语义一致。
    max_idx = min(args.idx + args.num, len(dataset))
    print("=" * 80, f"\nStart evaluation with {args.idx}~{max_idx} samples")

    scores_by_ratio = defaultdict(list)
    for data_idx in range(args.idx, max_idx):
        sample = dataset[data_idx]
        ctx_ids = model.encode(sample["prompt"])
        query_ids = model.apply_template(sample["query"])

        outputs = {}
        for t, ratio in enumerate(set_ratios()):
            kv = model.prefill(
                ctx_ids,
                prefill_chunk_size=args.prefill_chunk,
                window_size=args.window_size,
                chunk_ratio=ratio,
                level=args.level,
            )

            print(
                f"# prefill {model.name} mrcr-{data_idx}: "
                f"{len(ctx_ids[0])} tokens, KV cache {kv._mem()} GB, {kv.key_cache[0].dtype}"
            )

            response = model.generate(query_ids, kv=kv)
            score = grade(
                response, sample["answer"], sample["random_string_to_prepend"]
            )
            scores_by_ratio[ratio].append(score)

            outputs[ratio] = {
                "score": round(score, 4),
                "response": response,
                "ground-truth": sample["answer"],
                "n_tokens": sample["n_tokens"],
            }
            del kv

        save_result(model.name, args, outputs, data_idx)
        tt(f"[mrcr-{data_idx}]\n")

    print("\n" + "=" * 70)
    print(f"MRCR Evaluation Results (%) ({args.model}, {args.tag})")
    for ratio in sorted(scores_by_ratio.keys(), reverse=True):
        scores = scores_by_ratio[ratio]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"Ratio {ratio}: {avg_score * 100:.2f}")

    # 本地改动：原版不打印这行，而 scratch_repro_full.py 以日志出现 "Finished."
    # 作为作业成功的判据（与 eval.py/eval_chunk.py 一致）。
    print("Finished.")
