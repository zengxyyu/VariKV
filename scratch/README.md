# scratch/ — 工作产物归档

已完成阶段的日志、脚本和参考文件。**不是研究模块的一部分**，是上面各项结论的证据。
整理于 2026-08-03。

| 目录 | 内容 |
|---|---|
| `install/` | venv/CUDA 安装期日志与脚本，含 flash-attn 的 torch2.6 wheel（重建环境要用，见 CLAUDE.md「Environment」）和修好的 requirements |
| `repro_0730_qwen3/` | 2026-07-30 第一轮复现：Qwen3-8B，只跑 fastkvzip 一个方法，3 个数据集成功、`scbench_mf` 挂在 parquet 报错上。已被 Qwen2.5 的两轮取代，仅作 Figure 12 的参考点 |
| `repro_0731_qwen25/` | 2026-07-31 第二轮：Qwen2.5-7B-1M × 5 方法 × 3 数据集，2.9h 跑完。`fig11_results.log` 是解析出的表；`fig11_parse*.sh` 已被根目录的 `scratch_fig11_driver.sh` 取代 |
| `probe/` | 2 条样本的耗时探测。**投入 GPU-days 之前先跑一个**——成本无法从上下文长度推断 |
| `refs/` | `fastkvzip_paper.txt`：Fast KVzip 论文的文本提取版，本地唯一的论文副本。注意是从 PDF 提取的，图里的数据点已丢失 |

## 仍在根目录的活跃文件

2026-08-03 启动的 Figure 11 全量复现正在使用它们，跑完前**不要移动**——
调度器按相对路径调用脚本、bash 按字节偏移读取驱动脚本、两个日志是打开的 stdout 句柄：

- `scratch_repro_full.py` — 调度器（驱动脚本第 2 步会用相对路径调它跑 MRCR）
- `scratch_fig11_driver.sh` — 端到端驱动（等待 → MRCR → 解析 12 个数据集）
- `scratch_fig11_full_run.log` / `scratch_fig11_full_results.log` — 调度器与驱动的 stdout
- `scratch_repro_full_logs/` — 每个作业一个日志，外加 `.done__*` 完成标记

跑完后可整体归入 `scratch/repro_0803_fig11_full/`。
