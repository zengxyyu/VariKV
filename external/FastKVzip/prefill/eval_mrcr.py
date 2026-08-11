# ==============================================================================
# Benchmark evaluation without chunked-prefill-evict (MRCR)
#
# 本地新增文件（上游仓库没有）。用途：让 KVzip 基线能在 MRCR 上跑。
#
# 上游只提供 eval_chunk_mrcr.py（chunked-prefill-evict），对应 run.sh 里
# `python -B eval_chunk_mrcr.py -g fastkvzip -m $MODEL` 一条命令。但论文的 5 条曲线里，
# KVzip 基线在其余 11 个数据集上走的是 eval.py —— 即「不做 chunked-prefill-evict：
# 全量 prefill 后一次性打分，再按各比例剪枝」。若在 MRCR 上改用 eval_chunk_mrcr.py -g ""，
# 得到的是「KVzip 打分 + 分块驱逐」这一论文中不存在的配置，会让 MRCR 这一格
# 与其余 11 格设置不一致。本文件把 eval.py 的 unchunked 流程套到 MRCR 数据上，
# 保持 KVzip 基线在全部 12 个数据集上的设置统一。
#
# 与 eval.py 的对应关系：
#   - 强制 kv_type="retain"：RetainCache 只改 valid mask、不物理删 KV，
#     因此一次 prefill(do_score=True) 之后可以对同一份 cache 反复 prune 出 6 个比例。
#   - RetainCache.prune 每次都从原始 self.score 重算 mask（非累积剪枝），递减比例安全。
#   - model.generate 默认 update_cache=False，结尾 kv.slice 回滚生成产生的 KV，
#     所以同一份 cache 可以连续 generate 多次而不被污染。
# 与 eval_chunk_mrcr.py 的对应关系：
#   - 复用其 grade() 与 set_ratios()，outputs 结构逐字段对齐，
#     这样 results/parse_mrcr.py 能不加区分地解析两者的结果。
# ==============================================================================

from collections import defaultdict

from eval_chunk_mrcr import grade, set_ratios

if __name__ == "__main__":
    from args import args
    from model import ModelKVzip

    from data import load_dataset_all
    from utils import TimeStamp, save_result

    args.data = "mrcr"
    # 同 eval.py：只有带 gate 时才加 _w 后缀；KVzip（-g ""）的 tag 保持为空，
    # 结果目录落在 {idx}_qwen2.5-7b-instruct-1m，与它在其余 11 个数据集上的命名一致。
    if args.gate_path_or_name:
        args.tag += f"_w{args.window_size}"
    print(f"tag: {args.tag}")

    args.kv_type = "retain"
    model = ModelKVzip(args.model, args.kv_type, args.gate_path_or_name)
    dataset = load_dataset_all(args.data, model.tokenizer, n_data=2400)

    tt = TimeStamp(True)
    max_idx = min(args.idx + args.num, len(dataset))
    print("=" * 80, f"\nStart evaluation with {args.idx}~{max_idx} samples")

    scores_by_ratio = defaultdict(list)
    for data_idx in range(args.idx, max_idx):
        sample = dataset[data_idx]
        ctx_ids = model.encode(sample["prompt"])
        query_ids = model.apply_template(sample["query"])

        # 全量 prefill + 打分（gates is None 时 model.prefill 内部走 self.scoring，
        # 即 KVzip 的 context-reconstruction 注意力打分）
        kv = model.prefill(ctx_ids, do_score=True, window_size=args.window_size)
        print(
            f"# prefill {model.name} mrcr-{data_idx}: "
            f"{len(ctx_ids[0])} tokens, KV cache {kv._mem()} GB, {kv.key_cache[0].dtype}"
        )

        outputs = {}
        for ratio in set_ratios():
            kv.prune(ratio, args.level)
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

    print("Finished.")
