"""
阶段 1 生死实验的合成数据：可控干扰的 multi-key 检索。

设计动机（见 theory_distributional_memory.md §2.1）：
  固定更新率的点记忆在「保持」和「更新」两个子任务上不可能同时最优——
  率调高则记忆被后续噪声冲刷（保持失败），率调低则真正的改写写不进去（更新失败）。
  KL 自适应率的分布式记忆两边都能对。所以数据集必须**同时**包含两类查询，
  只考其中一类会给出假阴性（点记忆把固定率调对就能打平）。

两类查询：
  retain — 事实早出现，之后只有同格式干扰项，问它的值（考低 KL → 不更新 → 抗冲刷）
  update — 事实早出现，之后被真正改写，问最终值（考高 KL → 更新）

干扰强度轴 `n_distract`：needle 之后插入多少条同格式的无关 SET。
理论预测：干扰强度↑ → 分布式相对点记忆的优势↑；干扰≈0 时两者差异不大。
"""

import json
import random
from dataclasses import dataclass, asdict

# 值的词表：形容词-动物-两位数。固定词表 → 答案可精确匹配，且分词行为稳定。
ADJ = ["amber", "cobalt", "crimson", "jade", "ivory", "onyx", "scarlet", "azure",
       "golden", "silver", "violet", "umber", "teal", "rust", "slate", "coral"]
ANI = ["falcon", "otter", "lynx", "heron", "viper", "marten", "osprey", "badger",
       "raven", "shrike", "wolf", "ibis", "stoat", "grebe", "adder", "kite"]


def _mkval(rng):
    return f"{rng.choice(ADJ)}-{rng.choice(ANI)}-{rng.randint(10, 99)}"


def _mkkey(rng, used):
    while True:
        k = f"user_{rng.randint(1000, 9999)}"
        if k not in used:
            used.add(k)
            return k


@dataclass
class Sample:
    context: str
    question: str
    answer: str
    kind: str          # "retain" | "update"
    n_distract: int    # 干扰强度
    needle_depth: int  # needle 在多少条 SET 之后出现


def make_sample(rng, kind, n_distract, n_prefix=4):
    """造一条样本。

    结构：  [n_prefix 条前置干扰] [needle SET] [n_distract 条干扰]
            update 型会在干扰中段再插一条对同 key 的改写。
    """
    used = set()
    lines = []

    # 前置干扰：让 needle 不在最开头（避免模型靠位置偷答案）
    for _ in range(n_prefix):
        lines.append(f'SET {_mkkey(rng, used)} = "{_mkval(rng)}"')

    # needle
    key = _mkkey(rng, used)
    v1 = _mkval(rng)
    depth = len(lines)
    lines.append(f'SET {key} = "{v1}"')

    # 干扰项
    distract = [f'SET {_mkkey(rng, used)} = "{_mkval(rng)}"' for _ in range(n_distract)]

    if kind == "update":
        # 真正的改写插在干扰中段：模型必须更新记忆，不能只靠"抗覆盖"
        v2 = _mkval(rng)
        while v2 == v1:
            v2 = _mkval(rng)
        mid = len(distract) // 2
        distract.insert(mid, f'SET {key} = "{v2}"')
        answer = v2
    else:
        answer = v1

    lines.extend(distract)

    context = "[LOG]\n" + "\n".join(lines)
    question = f"What is the current value of {key}?"
    return Sample(context=context, question=question, answer=answer,
                  kind=kind, n_distract=n_distract, needle_depth=depth)


def build(n_per_cell, distract_levels, seed=0, kinds=("retain", "update")):
    """按 (kind × n_distract) 网格造数据集，每格 n_per_cell 条。"""
    rng = random.Random(seed)
    out = []
    for kind in kinds:
        for nd in distract_levels:
            for _ in range(n_per_cell):
                out.append(make_sample(rng, kind, nd))
    rng.shuffle(out)
    return out


def render(sample, tokenizer=None):
    """拼成送进 LLM 的完整 prompt。答案单独返回，供 teacher forcing / 精确匹配。"""
    prompt = (f"{sample.context}\n\n"
              f"[QUERY] {sample.question}\n"
              f"[ANSWER] ")
    return prompt, sample.answer


def save(samples, path):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


def load(path):
    with open(path) as f:
        return [Sample(**json.loads(line)) for line in f]


if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    # 先摸清 n_distract 与 token 长度的关系，好挑干扰强度档位
    print(f"{'n_distract':>11} {'chars':>8} {'tokens':>8}")
    for nd in [0, 50, 200, 800, 2000, 4000]:
        s = make_sample(random.Random(0), "retain", nd)
        p, a = render(s)
        print(f"{nd:>11} {len(p):>8} {len(tok(p)['input_ids']):>8}")

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        levels = [0, 200, 800, 2000]
        train = build(n_per_cell=400, distract_levels=levels, seed=0)
        val = build(n_per_cell=50, distract_levels=levels, seed=1234)
        save(train, "/home/ubuntu/zxy/vlm-memory/stage1/train.jsonl")
        save(val, "/home/ubuntu/zxy/vlm-memory/stage1/val.jsonl")
        print(f"\nwrote train={len(train)} val={len(val)} levels={levels}")
