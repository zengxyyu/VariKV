"""CPU 冒烟：合成 PrefSuf 任务的**结构**是否正确 —— 不加载模型。

判据先写死：
  ① 每个问句的 (前缀,后缀) 在**全部插入词**里恰好匹配 1 个 ⇒ 答案唯一；
  ② 同前缀的词数 = 1 + n_decoy，同后缀的词数 = 1 + n_decoy ⇒ 诱饵真的建起来了；
  ③ 词长 25、前后缀各 5、字母表与真实 PrefSuf 一致。
"""
import random
import re
import sys
sys.path.insert(0, "/home/ubuntu/zxy/vlm-memory")


import numpy as _np


class FakeTok:
    """真 tokenizer 的 encode 返回带 .tolist() 的张量，这里用 ndarray 顶替；
    返回字符码 ⇒ token 序列可逆回字符串，冒烟才能检查结构。"""
    def encode(self, t):
        return _np.array([[ord(c) % 1000 for c in t]])


import importlib.util                                              # noqa: E402
spec = importlib.util.spec_from_file_location(
    "advt", "/home/ubuntu/zxy/vlm-memory/scratch_adv_teacher.py")
# 直接 import 会触发 model 加载依赖，改成只把源码里的函数抠出来执行
src = open("/home/ubuntu/zxy/vlm-memory/scratch_adv_teacher.py").read()
i = src.index("def make_task_prefsuf")
j = src.index("# ────────────────────────────── 效用 J", i)
ns = {}
exec(src[i:j], ns)
make_task_prefsuf = ns["make_task_prefsuf"]

m = FakeTok()
ids = list(range(60000))                     # 假上下文
rng = random.Random(7)
N_FACT, N_DECOY = 8, 3
ctx, qas, meta = make_task_prefsuf(m, ids, 60000, 4096, N_FACT, rng, N_DECOY)
qas_dec = meta['qas_decoy']
print("meta ->", {k: v for k, v in meta.items() if k not in ("pos", "qas_decoy")})
print(f"上下文长度 {len(ids)} -> {len(ctx)}")

txt = "".join(chr(c) if 32 <= c < 127 else "?" for c in ctx)
words = re.findall(r"contains the word '([^']+)'", txt)
print(f"插入词 {len(words)} 个（应为 {N_FACT * (1 + 2 * N_DECOY)}）")
assert len(words) == N_FACT * (1 + 2 * N_DECOY), "插入词数不对"
assert all(len(w) == 25 for w in words), "词长不是 25"

qtxt = ["".join(chr(c) if 32 <= c < 127 else "?" for c in q) for q, _ in qas]
atxt = ["".join(chr(c) if 32 <= c < 127 else "?" for c in a).strip() for _, a in qas]
ok = True
for qi, (q, ans) in enumerate(zip(qtxt, atxt)):
    mm = re.search(r"prefix '([^']+)' and the suffix '([^']+)'", q)
    p_, s_ = mm.group(1), mm.group(2)
    both = [w for w in words if w.startswith(p_) and w.endswith(s_)]
    same_p = [w for w in words if w.startswith(p_)]
    same_s = [w for w in words if w.endswith(s_)]
    good = (len(both) == 1 and both[0] == ans
            and len(same_p) == 1 + N_DECOY and len(same_s) == 1 + N_DECOY)
    ok &= good
    if qi < 3 or not good:
        print(f"  q{qi} pre={p_} suf={s_}  两端都匹配={len(both)} "
              f"同前缀={len(same_p)} 同后缀={len(same_s)}  "
              f"答案对={both[:1] == [ans]}  {'OK' if good else '**FAIL**'}")
# ④ 自检⑤ 的诱饵：必须与正确答案同前缀、异后缀，且真的在插入词里
dec_ok = True
for qi, (q, ans) in enumerate(zip(qtxt, atxt)):
    mm = re.search(r"prefix '([^']+)' and the suffix '([^']+)'", q)
    p_, s_ = mm.group(1), mm.group(2)
    dq, da = qas_dec[qi]
    dqs = "".join(chr(c) if 32 <= c < 127 else "?" for c in dq)
    das = "".join(chr(c) if 32 <= c < 127 else "?" for c in da).strip()
    good = (dqs == q and das != ans and das.startswith(p_)
            and not das.endswith(s_) and das in words and len(das) == 25)
    dec_ok &= good
    if qi < 2 or not good:
        print(f"  诱饵 q{qi}: 问句相同={dqs == q} 同前缀={das.startswith(p_)} "
              f"异后缀={not das.endswith(s_)} 在词表里={das in words} "
              f"{'OK' if good else '**FAIL**'}")
ok &= dec_ok
print("\n判词：" + ("四条判据全过 —— 答案唯一、诱饵按 1+n_decoy 建起、词形对齐、自检⑤ 的诱饵同前缀异后缀。"
                    if ok else "**有 FAIL，不要跑 GPU**"))
print(f"\n样例问句： {qtxt[0].strip()}")
print(f"样例答案： {atxt[0]}")
print(f"样例诱饵（同前缀）： "
      f"{[w for w in words if w.startswith(qtxt[0].split(chr(39))[1])][:4]}")
