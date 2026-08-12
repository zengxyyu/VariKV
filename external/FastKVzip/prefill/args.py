import argparse

parser = argparse.ArgumentParser(description="")
# Method
parser.add_argument("-g", "--gate_path_or_name", type=str, default="fastkvzip")
parser.add_argument("--prefill_chunk", type=int, default=16000)
parser.add_argument("--window_size", type=int, default=4096)
parser.add_argument(
    "-r", "--ratio", type=float, default=0.3, help="compression ratio (= retained/full)"
)
parser.add_argument(
    "--kv_type",
    type=str,
    default="retain",
    choices=["evict", "retain"],
    help="retain: store full cache in storage, evict: delete the evicted KV from store.",
)
parser.add_argument(
    "--level",
    type=str,
    default="",
    choices=["pair", "pair-head", "pair-layer", "adakv-layer", ""],
    help="Eviction structure. pair-head/layer: uniform head/layer-budget; adakv-layer: with safeguard",
)
# Model and Data
parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M")
parser.add_argument(
    "-d",
    "--data",
    type=str,
    default="squad",
    help="check the dataset list in data/load.py (e.g., squad, scbench_kv)",
)
parser.add_argument("--idx", type=int, default=0, help="the index of a data example")
parser.add_argument(
    "--num", type=int, default=100, help="the total number of eval data"
)
parser.add_argument("--tag", type=str, default="", help="evaluation folder name tag")
# --- VariKV（本地新增，Stage 2b）---
# 给出 ckpt 即启用分布式记忆：驱逐决策仍完全由上游门控决定，
# 我们只接管「被驱逐的 KV 是丢弃还是吸收」，其余参数与基线逐字相同。
parser.add_argument("--varikv_ckpt", type=str, default="",
                    help="VariKV 记忆 checkpoint；给出则 kv_type 自动切到 memory")
parser.add_argument("--varikv_slots", type=int, default=16, help="每 head 的记忆条数 M")
# 默认 memory_retain：Figure 11 的基线全部跑在 kv_type="retain" 上（args.py 默认值，
# run.sh 不覆盖），所以方法也应挂在同一套机制上才逐项可比。
parser.add_argument("--varikv_kv_type", type=str, default="memory_retain",
                    choices=["memory", "memory_retain"])
# 消融：zero = 保持完全相同的驱逐与预算，但把读出的等效 KV 置零
parser.add_argument("--varikv_readout", type=str, default="normal",
                    choices=["normal", "zero"])
# 输出端门控残差：记忆不进 cache，改为 o = o_attn + sigmoid(g)·m(q)
parser.add_argument("--varikv_residual", action="store_true")
# --- gate surgery / 外科式消融（2026-08-12）：把 dist-vs-point 的四条通路拆开 ---
parser.add_argument("--varikv_gate_scale", type=float, default=1.0,
                    help="把残差注入幅度整体乘以此系数。用来测 point 的 14.60 是否"
                         "只是门开爆了（它学到 0.265，dist 只有 0.131）")
parser.add_argument("--varikv_gate_from", type=str, default="",
                    help="从另一个 ckpt 借 residual_gate（最锐利的对照：把 dist 的门"
                         "装到 point 身上，其余参数不变）")
parser.add_argument("--varikv_ablate", type=str, default="none",
                    choices=["none", "logvar", "precision", "eta"],
                    help="关掉 dist 的一条通路：logvar=读出不看方差；"
                         "precision=τ≡1；eta=写入强度换成本批均值（去内容相关性）")
# --- 质心臂（2026-08-12）：带计数的点质心 + 归一化感知读出，免训练 ---
# 与 --varikv_ckpt 互斥：它没有 ckpt，没有编码器，没有门。
parser.add_argument("--centroid_k", type=int, default=0,
                    help=">0 时启用质心读出，值为每 head 的簇数 K")
parser.add_argument("--centroid_rope", type=str, default="post",
                    choices=["post", "inv"],
                    help="post=直接平均 post-RoPE key（复现 E1b）；inv=逆旋到无位置帧（对照臂）")
args = parser.parse_args()


if args.level == "":
    # Use default eviction structure setting
    if "expect" in args.gate_path_or_name:
        args.level = "adakv-layer"
    elif "snap" in args.gate_path_or_name:
        args.level = "pair-head"
    else:
        args.level = "pair"

if args.tag:
    args.tag = f"_{args.tag}"

if args.gate_path_or_name:
    args.tag = "_" + args.gate_path_or_name.split("/")[-1] + args.tag
