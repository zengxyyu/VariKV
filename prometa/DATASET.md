# ProMeta 训练数据集 `manifest_v1_ss.jsonl` —— 构建配方与审计

`prometa_data/` 被 `.gitignore` 排除（单个 manifest 39 MB，存的是 token id）。
**本文件 + `prometa/dataset.py` + `scratch_prometa_selfstudy.py` + `prometa/merge_shards.py`
就是它的全部来源**；问句生成是 `do_sample=False` 的贪心解码，**可逐位复现**。

    sha256  0a677cada7b22497761746dd7fc39b5b5613c626ded803468019c730a0c31904

## 复现

```bash
# ① 骨架（上下文 + synth/continuation 的未来），源文档级 80/10/10 划分
.venv/bin/python -B prometa/dataset.py build --out prometa_data/manifest_v1.jsonl

# ② selfstudy 的 5 类问句（GPU，6 分片；--shard 按 id 取模 ⇒ 各片长度/划分构成均衡）
for i in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python -B scratch_prometa_selfstudy.py \
      --manifest prometa_data/manifest_v1.jsonl --shard $i --nshard 6 \
      --out prometa_data/ss_shard$i.jsonl &
done; wait

# ③ 合并 + 质量闸 + sha256
.venv/bin/python -B prometa/merge_shards.py prometa_data/manifest_v1.jsonl \
    "prometa_data/ss_shard*.jsonl" prometa_data/manifest_v1_ss.jsonl

# ④ 独立审计（只读 jsonl，不复用构建期任何状态）
.venv/bin/python -B scratch_prometa_audit_ds.py prometa_data/manifest_v1_ss.jsonl
```

## 组成（实测，非设计值）

| | 条数 | 源文档 | token |
|---|---|---|---|
| train | 160 | 383 | 5,700,977 |
| val | 20 | 47 | — |
| test | 20 | 48 | — |
| **合计** | **200** | **478** | **7,172,200** |

`split × kind`：train 104 selfstudy / 40 synth / 16 continuation；val 13/5/2；test 13/5/2。
`band`：8–16k ×70、16–32k ×60、32–64k ×40、64–128k ×30（三个 split 内分层一致）。
每条未来数 `M`：selfstudy 5、synth 5、**continuation 1**（续写只有一个真实未来）。

## 审计结论（`scratch_prometa_audit_ds.py`，10 条）

① 公共字段完整 ✓（`ss_ok` 只在 selfstudy、`meta` 只在 synth，按 kind 分是设计）
② futures 结构 / 非空 ✓（坏 future 0）　③ `n_ctx` 与 `ctx` 实长一致 ✓
④ **源文档级三向互不相交 ✓**（train∩val = train∩test = val∩test = 0）
⑤ 上下文全串与前 4000 token 都唯一 200/200 ✓（旧 `mix` 语料栽过前缀撞车）
⑥ 长度落在 band 内 ✓　⑦ token id 合法 ✓　⑧ 每条都有可用 chunk ✓（总 534，均 2.7）
⑨ **`a` 只有 synth 有（250/920）⇒ 只能 `--span q`**；训练脚本已加硬闸拒绝 `--span qa`
⑩ selfstudy 130/130 全部生成成功 ✓

**质量闸**（`prometa/merge_shards.py` 打印）：650 个问句，`grounded` 均 **0.893**
（<0.15 视为在编，最低 0.22，无一条低于 0.15）；同类型跨文档 Jaccard 均 **0.011**
（>0.6 视为退化成模板）：factual .011 / summary .008 / multihop .007 / method .004 /
comparison .023。

## 必须与数字一起引用的两条边界

1. **训练侧只有 423 个不同池化输入**（160 篇，`--n_chunk 8` 已取满；`--n_chunk 3`
   时只有 320）。Student 上下文路径 279,616 参数 ÷ 每输入 320 个输出数
   ⇒ **873 个输入才饱和**。423 < 873 ⇒ **容量上仍足以背下一张查找表**，
   所以 `--no_context` 与 `--shuffle_labels` 两个阴性对照**不是可选项**。
   旧语料只有 40 个输入，致盲对照拿到 +13.4% ≥ 看上下文的 +13.1%
   ⇒ 那 13% 全部来自一个近乎常数的预测器。
2. **`continuation` 的 M=1**：`match_loss(allow_extra=True)` 下只有 1 个 probe
   被监督，其余 4 个那一步只吃 diversity loss。可接受的理由是**另外 180 条
   记录的 M=5 会监督到全部 5 个**；若将来数据里 M=1 占比升高，这条理由失效。
