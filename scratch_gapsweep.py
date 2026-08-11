"""gap 目标 + 残差读出的三个 ckpt × Figure 11 其余 9 个数据集。

背景：这三个是当前最新的模型（obj=gap，带 residual_gate），此前只在
scbench_many_shot 和 scbench_kv 上评过。本脚本补齐其余 9 个数据集。

基线不重跑 —— scratch_stage2b_sweep.py 已经用 tag `_full` 跑过这 9 个数据集的
基线，配置逐项相同（prefill_chunk=16000, window=4096, num=100, level=pair,
比例走 set_ratios() 默认 5 档），与本次唯一的差别就是 --varikv_ckpt/--varikv_residual。

tag 逐 ckpt 区分：gap_fix03/dist 与 gap_rand/dist 都是 dist 模式，结果目录名只带
mode 不带 ckpt 名，同 tag 会互相覆盖。首字符不用下划线（否则目录出现双下划线，
手搓 -m 串会静默解析出 0 条）。tag 与正在跑的 scbench_kv 那批保持一致。

调度：长任务优先（实测单 config 时长 repoqa 5.83h ≫ choice_eng 0.44h），
makespan 才不会被尾部长任务拖长。GPU 0/1/2 上的 worker 会先等 scbench_kv
那批（scratch_gapstd_eval.sh）打出 ALL DONE 再领活，避免抢卡 OOM。

完成判定用 marker 文件（日志以 `Finished.` 结尾才写），不数结果文件：
choice_eng 只有 18 条、qa_eng 20 条、summary 70 条，按数量判定会永远重跑。
"""
import argparse
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFILL = ROOT / "external/FastKVzip/prefill"
PY = str(ROOT / ".venv/bin/python")
MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
LOGDIR = ROOT / "scratch_gapsweep_logs"
KV_STAMP = ROOT / "scratch_gapstd_eval.log"      # 等它出现 ALL DONE

# 实测单 config 时长（小时），长任务优先排
DATASETS = [
    "scbench_repoqa",          # 5.83
    "scbench_prefix_suffix",   # 3.23
    "scbench_mf",              # 2.97
    "scbench_vt",              # 2.20  ← 论文 Retr.MultiHop
    "scbench_summary",         # 1.88
    "gsm",                     # 1.19
    "scbench_qa_eng",          # 0.60
    "squad",                   # 0.55
    "scbench_choice_eng",      # 0.44
]
# (名字, tag, ckpt 相对 varikv/ 的路径)
CONFIGS = [
    ("gapf_dist",  "gfsd", "ckpt_gap_fix03/s2b_dist_k16.pt"),
    ("gapr_dist",  "grsd", "ckpt_gap_rand/s2b_dist_k16.pt"),
    ("gapr_point", "grsp", "ckpt_gap_rand/s2b_point_k16.pt"),
]


def marker(ds, name):
    return LOGDIR / f".done__{ds}__{name}"


def run_job(ds, name, tag, ckpt, gpu):
    log = LOGDIR / f"{ds}__{name}.log"
    cmd = [PY, "-B", "eval_chunk.py", "-g", "fastkvzip", "-m", MODEL,
           "-d", ds, "--tag", tag,
           "--varikv_ckpt", str(ROOT / "varikv" / ckpt),
           "--varikv_slots", "16", "--varikv_residual"]
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=PREFILL, stdout=f, stderr=subprocess.STDOUT,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
    ok = log.read_text(errors="ignore").rstrip().endswith("Finished.")
    if ok:
        marker(ds, name).touch()
    return ok


def kv_run_done():
    try:
        return "ALL DONE" in KV_STAMP.read_text(errors="ignore")
    except OSError:
        return True          # 没有那个文件 ⇒ 没在跑，直接开工


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[3, 4, 5, 6, 7],
                    help="立刻可用的卡")
    ap.add_argument("--wait_gpus", type=int, nargs="+", default=[0, 1, 2],
                    help="等 scbench_kv 那批跑完再加入的卡")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    LOGDIR.mkdir(parents=True, exist_ok=True)

    jobs = [(ds, n, tg, ck) for ds in DATASETS for n, tg, ck in CONFIGS
            if not marker(ds, n).exists()]
    print(f"待跑 {len(jobs)} 个任务；立即卡 {args.gpus}，等待卡 {args.wait_gpus}",
          flush=True)
    if args.plan:
        for j in jobs:
            print(f"  {j[0]:24s} {j[1]}")
        return

    lock, idx = threading.Lock(), [0]

    def worker(gpu, wait_first):
        if wait_first:
            while not kv_run_done():
                time.sleep(120)
            print(f"[gpu{gpu}] scbench_kv 那批已完成，加入", flush=True)
        while True:
            with lock:
                if idx[0] >= len(jobs):
                    return
                ds, name, tag, ck = jobs[idx[0]]
                idx[0] += 1
            t = time.time()
            print(f"[gpu{gpu}] START {ds}/{name}", flush=True)
            ok = run_job(ds, name, tag, ck, gpu)
            print(f"[gpu{gpu}] {'OK  ' if ok else 'FAIL'} {ds}/{name} "
                  f"{(time.time()-t)/60:.1f}min", flush=True)

    ths = ([threading.Thread(target=worker, args=(g, False)) for g in args.gpus]
           + [threading.Thread(target=worker, args=(g, True))
              for g in args.wait_gpus])
    [t.start() for t in ths]
    [t.join() for t in ths]
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
