"""Stage 2a：真实语料。fineweb-edu 长文档上的长上下文语言建模。

为什么先做这个而不是直接接 FastKVzip：完整的 per-head 集成里混着两个独立风险
（方法在真实文本上是否有效 / 变长布局移植是否正确），一起上就分不清是哪个坏了。
这里保持 varikv 现有的规整 per-token 布局，只把数据换成真实文本。

任务：把一篇文档预填进去（触发驱逐+吸收），在**末尾 held-out 的 target_len 个
token** 上算 nll。这是 Infini-attention 一类压缩记忆方法的标准评测。

与 stage1 的关键区别，以及由此带来的一个陷阱：
    stage1 的 needle 在最前面，所以训练时截断上下文的**尾部**（保住 needle）。
    真实文本的语言建模恰好相反 —— 紧挨 target 的那段文本是最强的预测信号，
    截尾会直接把任务毁掉。所以这里必须截**头部**，保留与 target 相邻的部分。
"""
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

CACHE_DIR = Path(__file__).resolve().parent.parent / "stage2_cache"


@dataclass
class RealSample:
    ids: List[int]          # 整篇文档的 token id
    n_tokens: int


def load_fineweb(
    tokenizer,
    n_train: int = 400,
    n_val: int = 60,
    min_len: int = 8000,
    max_len: int = 16000,
    seed: int = 0,
):
    """取指定长度区间的 fineweb-edu 文档，分词后缓存到磁盘。

    token_count 是 fineweb 自带的（别的分词器算的），只用来**粗筛**；
    真实长度以本地分词器为准，筛完再按实际长度过滤一次。
    """
    CACHE_DIR.mkdir(exist_ok=True)
    key = hashlib.md5(
        f"{tokenizer.name_or_path}|{n_train}|{n_val}|{min_len}|{max_len}|{seed}".encode()
    ).hexdigest()[:16]
    cache = CACHE_DIR / f"fineweb_{key}.pt"
    if cache.exists():
        d = torch.load(cache)
        return (
            [RealSample(**s) for s in d["train"]],
            [RealSample(**s) for s in d["val"]],
        )

    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        data_files="sample/10BT/000_00000.parquet",
        split="train",
    )
    length = np.array(ds.data.column("token_count"))
    idx = np.arange(len(length))[(length >= min_len) & (length < max_len)]
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    want = n_train + n_val
    out = []
    for i in idx:
        ids = tokenizer(ds[int(i)]["text"], add_special_tokens=True).input_ids
        if not (min_len <= len(ids) < max_len):     # 本地分词器口径下再过一遍
            continue
        out.append(RealSample(ids=ids, n_tokens=len(ids)))
        if len(out) >= want:
            break

    train, val = out[:n_train], out[n_train:want]
    torch.save(
        {"train": [s.__dict__ for s in train], "val": [s.__dict__ for s in val]}, cache
    )
    return train, val


def encode_real(sample: RealSample, device, target_len: int = 256, max_context: int = 0):
    """→ (ctx_ids, first_target_token, rest_target_tokens)

    max_context>0 时截断上下文的**头部** —— 与 stage1 相反，见模块 docstring。
    """
    ids = sample.ids
    ctx, tgt = ids[:-target_len], ids[-target_len:]
    if max_context and len(ctx) > max_context:
        ctx = ctx[-max_context:]
    t = lambda x: torch.tensor([x], dtype=torch.long, device=device)
    return t(ctx), t(tgt[:1]), t(tgt[1:])
