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
        # gate surgery / 消融。**必须进 tag**：否则不同配置写进同一结果目录互相覆盖
        # （2026-08-08 踩过：基线写进了 Figure-11 的目录，parse 把 54 条旧样本
        #  和 5 条新样本一起平均）。
        if args.varikv_gate_from:
            _g = _torch.load(args.varikv_gate_from, map_location=model.device)
            _sd = _g["memory"]
            assert "residual_gate" in _sd, f"{args.varikv_gate_from} 没有 residual_gate"
            with _torch.no_grad():
                _mem.residual_gate.copy_(_sd["residual_gate"].to(_mem.residual_gate))
            args.tag += "_gfrom" + _g.get("mode", "?")
            print(f"[VariKV] 借用 {args.varikv_gate_from} 的门 "
                  f"σ mean={_torch.sigmoid(_sd['residual_gate']).mean():.4f}")
        if args.varikv_gate_scale != 1.0:
            args.tag += f"_gs{args.varikv_gate_scale:g}".replace(".", "p")
        if args.varikv_ablate != "none":
            setattr(_mem, f"ablate_{args.varikv_ablate}"
                    if args.varikv_ablate != "logvar" else "ablate_logvar_read", True)
            args.tag += f"_ab{args.varikv_ablate}"
            print(f"[VariKV] 消融 {args.varikv_ablate}")
        model.varikv_gate_scale = args.varikv_gate_scale
        model.varikv_inv_freq = _rot.inv_freq.detach().clone() if _rot else None
        print(f"[VariKV] loaded {args.varikv_ckpt} mode={_ck.get('mode')} "
              f"M={args.varikv_slots}")
    elif args.ctrlm_ckpt:
        # VariKV-B 最终版：学出来的历史控制状态
        import sys as _sys, torch as _torch
        _sys.path.insert(0, "/home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill")
        from attention.control_memory import ControlMemory as _CM

        args.kv_type = "control_learned"
        _ck = _torch.load(args.ctrlm_ckpt, map_location="cpu")
        _mode = args.ctrlm_mode or _ck.get("mode", "stateful")
        args.tag += f"_ctrlm{_mode[:4]}{_ck.get('slots', args.ctrlm_slots)}"
        model = ModelKVzip(args.model, args.kv_type, args.gate_path_or_name)
        _arch = _ck.get("arch", "memory")
        _ns = _ck.get("slots", args.ctrlm_slots)
        # **typed 从权重形状推断，别依赖构造函数默认值。** typed=True 时状态按动作
        # 分型（前 K 槽=保留史、后 K 槽=驱逐史），M_init 是 2K 槽；默认值改过一次，
        # 靠默认值加载会在某天静默错配。
        if _arch == "memory":
            _typed = _ck["state"]["M_init"].shape[2] == 2 * _ns
            _cls, _kw = _CM, {"typed": _typed}
        else:
            from attention.calib_scorer import CalibScorer as _CS
            _typed, _cls, _kw = None, _CS, {"arch": _arch}
            _mode = "memoryless"            # CalibScorer 无记忆
            args.tag += f"_{_arch}"
        _cm = _cls(_ck.get("d_kv", 128), _ck["L"], _ck["H"], n_slots=_ns,
                   d_m=_ck.get("dim", args.ctrlm_dim), mode=_mode, **_kw)
        _cm.load_state_dict(_ck["state"])
        if args.ctrlm_alpha >= 0:
            import math as _math
            with _torch.no_grad():
                _p = min(max(args.ctrlm_alpha / _cm.alpha_max, 1e-6), 1 - 1e-6)
                _cm.alpha_on.fill_(_math.log(_p / (1 - _p)))
            args.tag += f"_a{args.ctrlm_alpha:g}"
        model.ctrl_module = _cm.to(model.device).eval()
        model.ctrl_seed = args.ctrl_seed
        print(f"[CtrlM] {args.ctrlm_ckpt} mode={_mode} slots={_ns} "
              f"typed={_typed} alpha={float(_cm.alpha):.4f} "
              f"params={_cm.n_params()/1e3:.1f}K")
    elif args.ctrl:
        # 控制臂：只改 kv_type + 几个标量，其余评测参数与基线逐字一致。
        args.kv_type = "control"
        args.tag += f"_ctrl{args.ctrl_src[:3]}{args.ctrl_beta:g}"
        if args.ctrl_beta_group != 0.0:
            args.tag += f"_g{args.ctrl_beta_group:g}"
        if args.ctrl_feat != "key":
            args.tag += f"_{args.ctrl_feat}"
        if args.ctrl_rho != 1.0:
            args.tag += f"_rho{args.ctrl_rho:g}"
        if args.ctrl_shuffle:
            args.tag += "_shuf"
        if args.ctrl_rope != "post":
            args.tag += f"_{args.ctrl_rope}"
        model = ModelKVzip(args.model, args.kv_type, args.gate_path_or_name)
        for k in ("beta", "beta_group", "rho", "src", "feat", "rope",
                  "shuffle", "seed"):
            setattr(model, f"ctrl_{k}", getattr(args, f"ctrl_{k}"))
        if args.ctrl_rope == "inv":
            _rot = getattr(model.model.model, "rotary_emb", None)
            assert _rot is not None, "inv 模式需要 rotary_emb"
            model.varikv_inv_freq = _rot.inv_freq.detach().clone()
        print(f"[Ctrl] beta={args.ctrl_beta} src={args.ctrl_src} feat={args.ctrl_feat} "
              f"rho={args.ctrl_rho} shuffle={args.ctrl_shuffle}")
    elif args.centroid_k > 0:
        # 免训练的质心臂：只改 kv_type + K，其余评测参数与基线逐字一致。
        args.kv_type = "centroid"
        args.tag += f"_cen{args.centroid_k}"
        if args.centroid_rope != "post":
            args.tag += f"_{args.centroid_rope}"
        model = ModelKVzip(args.model, args.kv_type, args.gate_path_or_name)
        model.varikv_K = args.centroid_k
        model.varikv_rope_mode = args.centroid_rope
        if args.centroid_rope == "inv":
            _rot = getattr(model.model.model, "rotary_emb", None)
            assert _rot is not None, "inv 模式需要 rotary_emb"
            model.varikv_inv_freq = _rot.inv_freq.detach().clone()
        print(f"[Centroid] K={args.centroid_k}/head rope={args.centroid_rope}")
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
