# ==============================================================================
# Benchmark evaluation with chunked-prefill-evict
# ==============================================================================

from collections import defaultdict

from eval import get_data_list, set_ratios

if __name__ == "__main__":
    from args import args
    from model import ModelKVzip

    from data import DataWrapper, load_dataset_all
    from utils import Evaluator, TimeStamp, save_result, set_gen_length

    args.tag += f"_chunk{args.prefill_chunk//1000}k_w{args.window_size}"
    print(f"tag: {args.tag}")

    # --- VariKV 注入（本地新增）：除 kv_type 外所有评测参数与基线完全一致 ---
    if args.varikv_ckpt:
        import sys as _sys, torch as _torch
        _sys.path.insert(0, "/home/ubuntu/zxy/vlm-memory")
        from varikv.config import Config as _Cfg
        from varikv.memory import DistributionalMemory as _DM

        args.kv_type = args.varikv_kv_type
        model = ModelKVzip(args.model, args.kv_type, args.gate_path_or_name)
        _ck = _torch.load(args.varikv_ckpt, map_location=model.device)
        # mode 必须进 tag：否则 dist 与 point 写进同一个结果目录互相覆盖
        args.tag += f"_varikv{_ck.get('mode','dist')}{args.varikv_slots}"
        _cfg = _Cfg(); _cfg.memory.num_slots = args.varikv_slots
        _H = model.config.num_key_value_heads
        _L = model.config.num_hidden_layers
        _hd = getattr(model.config, "head_dim",
                      model.config.hidden_size // model.config.num_attention_heads)
        # n_groups 决定是否创建 residual_gate。残差模式的 ckpt 里带这个键，
        # 不传 n_groups 就会 "Unexpected key(s): residual_gate" 直接崩。
        _mem = _DM(2 * _hd, _cfg.memory, mode=_ck.get("mode", "dist"),
                   n_groups=(_L * _H) if args.varikv_residual else 0).to(
            model.device, dtype=_torch.float32)
        _miss = _mem.load_state_dict(_ck["memory"], strict=False)
        if _miss.missing_keys or _miss.unexpected_keys:
            print(f"[VariKV] 缺失键 {list(_miss.missing_keys)} "
                  f"多余键 {list(_miss.unexpected_keys)}")
        _mem.eval()
        _mem.reset(1, _L * _H, device=model.device, dtype=_torch.float32)
        model.varikv_memory = _mem
        model.varikv_M = args.varikv_slots
        model.varikv_readout = args.varikv_readout
        model.varikv_residual = args.varikv_residual
        if args.varikv_residual:
            args.tag += "_res"
        if args.varikv_readout != "normal":
            args.tag += f"_ro{args.varikv_readout}"
        _rot = getattr(model.model.model, "rotary_emb", None)
        model.varikv_inv_freq = _rot.inv_freq.detach().clone() if _rot else None
        print(f"[VariKV] loaded {args.varikv_ckpt} mode={_ck.get('mode')} "
              f"M={args.varikv_slots}")
    else:
        model = ModelKVzip(args.model, args.kv_type, args.gate_path_or_name)

    for args.data in get_data_list(args.data, model.name):
        dataset = load_dataset_all(args.data, model.tokenizer)  # list of data
        dataset = DataWrapper(args.data, dataset, model)
        set_gen_length(args.data, model)

        tt = TimeStamp(True)
        max_idx = min(args.idx + args.num, len(dataset))
        print("=" * 80, f"\nStart evaluation with {args.idx}~{max_idx} samples")

        for data_idx in range(args.idx, max_idx):
            # Get full KV cache generation results
            kv = dataset.prefill_context(data_idx, do_score=False)
            inputs, info = dataset.generate_answer(data_idx, kv, prob=False)
            eval = Evaluator(model, inputs, info)
            del kv

            outputs = defaultdict(list)
            for t, ratio in enumerate(set_ratios()):
                # Get generation results with chunked-prefill-evict
                kv = dataset.prefill_context(
                    data_idx,
                    prefill_chunk=args.prefill_chunk,
                    window_size=args.window_size,
                    chunk_ratio=ratio,
                    level=args.level,
                )
                results = eval(kv, generate=True)

                for fmt, v in results.items():
                    outputs[fmt].append([[ratio, 0, 0], v])

                del kv

            save_result(model.name, args, outputs, data_idx)

            tt(f"[{args.data}-{data_idx}]\n")
            del inputs, info, eval
        print("Finished.")
