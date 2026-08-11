"""验证根因：'[ANSWER] ' 末尾空格造成 prompt 与 answer 的分词错位。

同时确认这个 bug 也污染了训练（forward_loss 的 teacher forcing 用同一个 q_text/a_ids）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
ans = "crimson-kite-33"

print("=" * 78)
print("当前实现：q_text 以空格结尾，answer 单独分词")
print("=" * 78)
q_sp = "\n[ANSWER] "
q_ids = tok(q_sp, add_special_tokens=False).input_ids
a_ids = tok(ans, add_special_tokens=False).input_ids
print(f"  q_text  {q_sp!r}")
print(f"    ids   {q_ids} -> {[tok.decode([i]) for i in q_ids]}")
print(f"  answer  {ans!r}")
print(f"    ids   {a_ids} -> {[tok.decode([i]) for i in a_ids]}")
print(f"  拼接序列 = {q_ids + a_ids}")

print("\n" + "=" * 78)
print("自然文本：整体分词（模型真正会生成的 token 序列）")
print("=" * 78)
joint = tok(q_sp + ans, add_special_tokens=False).input_ids
print(f"  {q_sp + ans!r}")
print(f"    ids   {joint} -> {[tok.decode([i]) for i in joint]}")

print("\n  两者相同？", q_ids + a_ids == joint)
print(f"  分叉点：拼接={q_ids + a_ids}\n           自然={joint}")

print("\n" + "=" * 78)
print("修复方案：q_text 不带尾空格，answer 带前导空格")
print("=" * 78)
q_ns = "\n[ANSWER]"
q2 = tok(q_ns, add_special_tokens=False).input_ids
a2 = tok(" " + ans, add_special_tokens=False).input_ids
joint2 = tok(q_ns + " " + ans, add_special_tokens=False).input_ids
print(f"  q_text  {q_ns!r} -> {[tok.decode([i]) for i in q2]}")
print(f"  answer  {' ' + ans!r} -> {[tok.decode([i]) for i in a2]}")
print(f"  拼接 == 自然？ {q2 + a2 == joint2}")
