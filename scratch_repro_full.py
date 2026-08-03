#!/usr/bin/env python3
"""
完整复现 Fast KVzip 论文：6 模型 × 5 方法 × 11 数据集（+ MRCR + math）。

调度器：8 张 H100，每张卡一个 worker，从优先级队列取 (model, method, dataset) 作业。
断点续跑：作业开跑前检查结果文件是否已齐，齐了就跳过。
优先级：先出招牌行（Qwen3-8B × fastkvzip），再补基线方法，再补其余模型；
        每个 (model,method) 内部按数据集长度从短到长，让早期就有可解析的结果。

用法：
  python scratch_repro_full.py --plan          # 只打印作业计划和进度，不跑
  python scratch_repro_full.py --run           # 开跑
  python scratch_repro_full.py --run --models Qwen/Qwen3-8B   # 只跑指定模型
"""

import argparse
import os
import queue
import subprocess
import threading
import time

ROOT = "/home/ubuntu/zxy/vlm-memory"
PREFILL = f"{ROOT}/external/FastKVzip/prefill"
PY = f"{ROOT}/.venv/bin/python"
LOGDIR = f"{ROOT}/scratch_repro_full_logs"
NUM_GPUS = 8
NUM_EXAMPLES = 100

# 论文里的模型（gate 权重均已发布，自动下载）。
# 顺序 = 复现优先级：Qwen2.5-7B-1M 必须第一，它才是 Figure 11（主结果图，逐任务×逐比例）
# 用的模型，也是 run.sh 的默认。其余四个是 Figure 12（跨模型泛化，12 数据集平均）的面板。
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct-1M",   # Figure 11 主结果
    "Qwen/Qwen3-8B",                 # Figure 12
    "google/gemma-3-12b-it",         # Figure 12（滑窗混合注意力，只压 global）
    "Qwen/Qwen2.5-14B-Instruct-1M",  # Figure 12
    "Qwen/Qwen3-8B-FP8",             # Figure 12（动态 FP8）
    "Qwen/Qwen3-14B",                # 正文提及
]

# run.sh 里的 5 个方法： (标签, 脚本, -g 参数, level)
# level 决定结果文件名 output-{level}.json，见 args.py 的默认值推导
METHODS = [
    ("fastkvzip", "eval_chunk.py", "fastkvzip", "pair"),        # 本文方法
    ("kvzip", "eval.py", "", "pair"),                            # KVzip（不分块预填）
    ("duoattn", "eval_chunk.py", "head", "pair"),                # DuoAttention
    ("expected", "eval_chunk.py", "expect", "adakv-layer"),      # Expected Attention
    ("snapkv", "eval_chunk.py", "snap", "pair-head"),            # SnapKV
]

# -d all = long + short + mid，按上下文长度从短到长排（短的先出结果）。
# 长度取自 data/load.py:get_data_list 的注释，是 **Qwen2.5-7B-1M 下的真实值**：
# 该模型不触发 _short/_mid 替换，跑的是全长版本，比 Qwen3 的替换版贵一个量级
# （如 scbench_kv：Qwen3 走 _short 约 20k，这里是 169k）。
DATASETS = [
    "gsm", "squad",                                    # 86 / 203 tokens
    "scbench_many_shot",                               # 26k
    "scbench_repoqa",                                  # 72k
    "scbench_qa_eng", "scbench_choice_eng",            # 122k / 119k
    "scbench_prefix_suffix", "scbench_summary",        # 113k / 118k
    "scbench_vt",                                      # 125k
    "scbench_mf",                                      # 150k
    "scbench_kv",                                      # 169k
    # Figure 11 的第 12 个数据集。上下文 ≤128K，走专用脚本（见 run_job）：
    # 数据格式不经 DataWrapper，打分用 SequenceMatcher，结果由 results/parse_mrcr.py 解析。
    "mrcr",
]

# mrcr 的数据装载/打分逻辑与其余数据集不同，需换用专用脚本。
# eval_chunk_mrcr.py 对应 eval_chunk.py（分块预填驱逐），
# eval_mrcr.py 是本地新增的 eval.py 对应物（不分块，供 KVzip 基线用）。
MRCR_SCRIPT = {"eval_chunk.py": "eval_chunk_mrcr.py", "eval.py": "eval_mrcr.py"}


def resolve_dataset(name, model):
    """复刻 eval.py:get_data_list 的模型相关改名，好做结果查重。"""
    m = model.lower()
    if not any(k in m for k in ("qwen3", "gemma3", "gemma-3")):
        return name
    if name == "scbench_prefix_suffix":
        return "scbench_prefix_suffix_short"
    if "instruct" not in m:
        if name == "scbench_kv":
            return "scbench_kv_short"
        if name == "scbench_mf":
            return "scbench_mf_mid"
    return name


def model_shortname(model):
    """复刻 ModelKVzip 的 name（结果目录里用的那个）。"""
    return model.split("/")[-1].lower()


def result_tag(gate, script):
    """复刻 args.py + eval*.py 的 tag 拼接。"""
    tag = f"_{gate.split('/')[-1]}" if gate else ""
    if script == "eval_chunk.py":
        tag += "_chunk16k_w4096"
    elif gate:  # eval.py 只在有 gate 时加 _w
        tag += "_w4096"
    return tag


def marker_path(model, method, dataset):
    label = method[0]
    return f"{LOGDIR}/.done__{model_shortname(model)}__{label}__{resolve_dataset(dataset, model)}"


def is_done(model, method, dataset):
    """完成判定以标记文件为准。

    不能用「结果文件数 == 100」判定：有的数据集本身不足 100 条
    （scbench_many_shot 只有 54 条），那样会永远判为未完成、无限重跑。
    标记文件在作业跑完且日志出现 "Finished." 时写入。
    """
    label, script, gate, level = method
    data = resolve_dataset(dataset, model)
    tag = result_tag(gate, script)
    name = model_shortname(model)
    have = sum(os.path.exists(f"{PREFILL}/results/{data}/{i}_{name}{tag}/output-{level}.json")
               for i in range(NUM_EXAMPLES))
    return os.path.exists(marker_path(model, method, dataset)), have


def build_jobs(models, datasets, methods):
    jobs = []
    sel = [m for m in METHODS if m[0] in methods]
    for model in models:
        for method in sel:
            for dataset in datasets:
                jobs.append((model, method, dataset))
    return jobs


def run_job(job, gpu, logf):
    model, method, dataset = job
    label, script, gate, level = method
    if dataset == "mrcr":
        # 两个 mrcr 脚本都把 args.data 硬编码成 "mrcr"，-d 传什么都会被覆盖，
        # 但保留 -d 以免 argparse 行为随上游变动而不同。
        script = MRCR_SCRIPT[script]
    cmd = [PY, "-B", script, "-m", model, "-d", dataset, "--num", str(NUM_EXAMPLES)]
    if gate:
        cmd += ["-g", gate]
    else:
        cmd += ["-g", ""]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    with open(logf, "w") as f:
        return subprocess.call(cmd, cwd=PREFILL, env=env, stdout=f, stderr=subprocess.STDOUT)


def worker(gpu, q, state, lock):
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        model, method, dataset = job
        label = method[0]
        key = f"{model_shortname(model)}__{label}__{resolve_dataset(dataset, model)}"
        done, have = is_done(model, method, dataset)
        if done:
            with lock:
                state["skipped"] += 1
                print(f"[gpu{gpu}] SKIP {key} (already complete)", flush=True)
            q.task_done()
            continue
        logf = f"{LOGDIR}/{key}.log"
        with lock:
            print(f"[gpu{gpu}] START {key} (have {have}/{NUM_EXAMPLES})", flush=True)
        t0 = time.time()
        rc = run_job(job, gpu, logf)
        dt = time.time() - t0
        # 跑完的判据：进程正常退出 且 日志出现 "Finished."（eval 脚本收尾时打的）
        finished = False
        if rc == 0:
            try:
                with open(logf) as f:
                    finished = "Finished." in f.read()[-4000:]
            except OSError:
                pass
        if finished:
            open(marker_path(model, method, dataset), "w").close()
        _, have2 = is_done(model, method, dataset)
        with lock:
            state["done" if finished else "failed"] += 1
            status = "OK" if finished else f"INCOMPLETE({have2} results) rc={rc}"
            print(f"[gpu{gpu}] END   {key} {status} {dt/60:.1f}min", flush=True)
        q.task_done()


def main():
    global NUM_EXAMPLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--datasets", nargs="*", default=DATASETS,
                    help="只跑这些数据集（默认 -d all 的 11 个）")
    ap.add_argument("--methods", nargs="*", default=[m[0] for m in METHODS],
                    help="只跑这些方法标签")
    ap.add_argument("--gpus", nargs="*", type=int, default=list(range(NUM_GPUS)),
                    help="用哪几张卡（默认全部）")
    ap.add_argument("--num", type=int, default=NUM_EXAMPLES,
                    help="每个数据集评测多少条（论文用 100）")
    args = ap.parse_args()
    NUM_EXAMPLES = args.num

    os.makedirs(LOGDIR, exist_ok=True)
    jobs = build_jobs(args.models, args.datasets, args.methods)

    if args.plan or not args.run:
        ndone = sum(is_done(*j)[0] for j in jobs)
        print(f"total jobs: {len(jobs)}  complete: {ndone}  remaining: {len(jobs)-ndone}")
        print(f"models={len(args.models)} methods={len(args.methods)} "
              f"datasets={len(args.datasets)} num={NUM_EXAMPLES} gpus={args.gpus}")
        for j in jobs:
            done, have = is_done(*j)
            if not done:
                print(f"  TODO {model_shortname(j[0])} / {j[1][0]} / "
                      f"{resolve_dataset(j[2], j[0])}  (have {have})")
        if not args.run:
            return

    q = queue.Queue()
    for j in jobs:
        q.put(j)
    state = {"done": 0, "failed": 0, "skipped": 0}
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(g, q, state, lock), daemon=True)
               for g in args.gpus]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"\nALL DONE in {(time.time()-t0)/3600:.1f}h  {state}", flush=True)


if __name__ == "__main__":
    main()
