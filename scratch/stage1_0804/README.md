# Stage-1 首次端到端运行（2026-08-04）—— 已作废，仅留作证据

跑于 2026-08-04 01:21→02:03，`scratch_stage1_driver.sh`，`steps=1500 budget=256 eval_limit=120`。

**为什么作废**：当时 prompt 以 `"[ANSWER] "` 结尾、答案单独分词，在 Qwen BPE 下
`[..., ']', ' ', 'cr', 'imson']` 与自然文本的 `[..., ']', ' crimson']` 不一致。
后果有两层：

- 评测五档全 `0.000`（`eval_all.log`），且 tier 1（无记忆参与）和 `n_distract=0`
  （109 token，不触发驱逐）也是 0 —— 说明是评测链路坏了，不是方法结论。
- **训练目标同样失真**：`forward_loss` 用同一套 `q_ids`+`a_ids` 拆分，
  所以这里的三个 ckpt 是在一个模型基本不会产生的 token 序列上训出来的。

因此 `tier{2,4,5}.pt` **不能与修复后的 ckpt 直接比较**。
`train_tier*.log` 里那组 lm_loss（tier5 2.25 < tier4 2.77 < tier2 3.98）是在
失真目标下测得的，有参考价值但不是结论。

根因分析与修复见 `CLAUDE.md` 的 "Root cause found and fixed (2026-08-07)" 一节。
