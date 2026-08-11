"""Stage 2b 全量评测调度器：Figure 11 的数据集 × {基线, dist, point}。

**所有评测参数一律走 args.py 默认值**，与 run.sh 逐字一致：
    prefill_chunk=16000, window_size=4096, num=100, ratio 序列由 set_ratios() 给,
    level 由 fastkvzip 门控推导为 pair。
命令行里除 `-d`、`--tag` 外，只多一个 `--varikv_ckpt` —— 这是与基线的唯一差别。

完成判定用 marker 文件（日志以 `Finished.` 结尾才写），不数结果文件：
several 数据集的样本数少于 --num 100（choice_eng 18 / qa_eng 20 / many_shot 54 …），
按数量判定会永远重跑 —— 这个坑 scratch_repro_full.py 已经踩过一次。
"""
import argparse
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFILL = ROOT / "external/FastKVzip/prefill"
PY = str(ROOT / ".venv/bin/python")
MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
CKDIR = ROOT / "varikv/ckpt_stage2b_matched"
LOGDIR = ROOT / "scratch_stage2b_logs/sweep"
TAG = "_full"

# mrcr 不在此列：它走专用脚本 eval_chunk_mrcr.py，我们的注入还没接到那条路上。
DATASETS = [
    "squad", "gsm",                                    # 203 / 86 tokens，最便宜
    "scbench_choice_eng", "scbench_qa_eng",            # 18 / 20 条样本
    "scbench_repoqa",                                  # 72k
    "scbench_prefix_suffix", "scbench_summary",        # 113k / 118k
    "scbench_vt", "scbench_mf",                        # 125k / 150k
]
CONFIGS = [
    ("baseline", None),
    ("dist", CKDIR / "s2b_dist_k16.pt"),
    ("point", CKDIR / "s2b_point_k16.pt"),
]


def marker(ds, name):
    return LOGDIR / f".done__{ds}__{name}"


def run_job(ds, name, ckpt, gpu):
    log = LOGDIR / f"{ds}__{name}.log"
    cmd = [PY, "-B", "eval_chunk.py", "-g", "fastkvzip", "-m", MODEL,
           "-d", ds, "--tag", TAG]
    if ckpt:
        cmd += ["--varikv_ckpt", str(ckpt)]
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=PREFILL, stdout=f, stderr=subprocess.STDOUT,
                       env={**__import__("os").environ,
                            "CUDA_VISIBLE_DEVICES": str(gpu)})
    ok = log.read_text(errors="ignore").rstrip().endswith("Finished.")
    if ok:
        marker(ds, name).touch()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 6, 7])
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    LOGDIR.mkdir(parents=True, exist_ok=True)

    jobs = [(ds, n, c) for ds in DATASETS for n, c in CONFIGS
            if not marker(ds, n).exists()]
    print(f"待跑 {len(jobs)} 个任务，GPU {args.gpus}")
    for j in jobs:
        print(f"  {j[0]:24s} {j[1]}")
    if args.plan:
        return

    lock, idx = threading.Lock(), [0]

    def worker(gpu):
        while True:
            with lock:
                if idx[0] >= len(jobs):
                    return
                ds, name, ck = jobs[idx[0]]
                idx[0] += 1
            t = time.time()
            print(f"[gpu{gpu}] START {ds}/{name}", flush=True)
            ok = run_job(ds, name, ck, gpu)
            print(f"[gpu{gpu}] {'OK  ' if ok else 'FAIL'} {ds}/{name} "
                  f"{(time.time()-t)/60:.1f}min", flush=True)

    ths = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    [t.start() for t in ths]
    [t.join() for t in ths]
    print("ALL DONE")


if __name__ == "__main__":
    main()
