"""应用 P0-A：空记忆不得注入（NEXT_STEPS.md v5 §4 P0-A）。

`attn.py:149` 无条件调用 `memory_residual`，所以在第一次吸收发生之前，槽还是初值时也会往
注意力输出里注入东西。证据：同一 ckpt 跨独立 job 的 ratio-1.0 分数逐字相同、不同 ckpt 之间
不同（68.20/66.80/68.60/67.20/67.80/70.40），说明 ckpt 决定了本该与记忆无关的那一档分数。

修法：在 `memory_residual` 开头加一个 guard，未吸收过任何东西就返回全零，
形状与正常返回一致（[B, T, H*Gq*d]）。

幂等：已打过补丁就跳过。
"""
import pathlib
import re
import sys

P = pathlib.Path("/home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill/attention/memcache_retain.py")
src = P.read_text()

MARK = "P0-A guard"
if MARK in src:
    print("已打过补丁，跳过"); sys.exit(0)

# 在 memory_residual 的 docstring 结束后插入 guard
m = re.search(r'(    def memory_residual\(self, query_states, layer_idx\):\n        """.*?"""\n)',
              src, re.S)
if not m:
    print("找不到 memory_residual 的 docstring，未修改"); sys.exit(1)

guard = '''        # ---- P0-A guard（2026-08-11）：空记忆不得注入 ----
        # 本函数在 attn.py:149 被**无条件**调用，因此第一次吸收发生之前（槽仍是初值）
        # 也会往注意力输出里加东西。实测后果：同一 ckpt 跨独立 job 的 ratio-1.0 分数
        # 逐字相同、不同 ckpt 之间不同（68.20/66.80/68.60/67.20/67.80/70.40），
        # 即 ckpt 决定了本该与记忆无关的那一档分数 ⇒ full-cache 参照被污染。
        # 未吸收过任何东西时返回全零，形状与正常返回一致。
        if getattr(self, "_absorbed_upto", 0) <= 0:
            H = self.n_heads_kv
            B, HQ, T, d = query_states.shape
            return query_states.new_zeros(B, T, HQ * d)
'''
src = src[:m.end(1)] + guard + src[m.end(1):]
P.write_text(src)
print("补丁已应用")
