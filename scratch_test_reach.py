#!/usr/bin/env python3
"""slack 可达性证书的单元测试 —— 手工构造四类例子，零 GPU、零依赖。

**为什么必须有**：这个证书现在是论文的理论核心之一
（74/74 chunk 不可达 ⇒ `Q_current ⊊ Q_feasible`），
而它此前**没有任何测试**。判据是

    slack(q) = min_h [ s_{h,(q_h)} + a_h ] − max_h [ s_{h,(q_h+1)} − a_h ]
    slack > 0  ⟺  存在公共阈值 τ′ 使目标配额 q 可表示

四类必测：可达 / 不可达 / 边界 `q=0` 与 `q=n` / 平局。
**还要测一条我推出来的结论**：有界平移与有界逐 token 修正**可达集相同**
（因为配额只取决于边界那一对），所以两者的 slack 判据应当逐位一致。
"""
import numpy as np

INF = np.inf


def slack(S, a, q):
    """S: list of 每头降序分数；a: 每头界；q: 每头目标配额。"""
    hi = INF; lo = -INF
    for sh, ah, qh in zip(S, a, q):
        n = len(sh)
        s_q  = sh[qh-1] if qh >= 1 else  INF     # 第 q 名（要留）
        s_q1 = sh[qh]   if qh < n    else -INF   # 第 q+1 名（要删）
        hi = min(hi, s_q + ah)
        lo = max(lo, s_q1 - ah)
    return hi - lo


def brute(S, a, q, grid=20001):
    """暴力：在细网格上找 τ′，逐头检查最有利取值能否达成 q。"""
    allv = np.concatenate([np.asarray(x) for x in S])
    lo, hi = allv.min()-max(a)-1, allv.max()+max(a)+1
    for tau in np.linspace(lo, hi, grid):
        ok = True
        for sh, ah, qh in zip(S, a, q):
            n = len(sh)
            up = (sh[qh-1] + ah) if qh >= 1 else INF      # 第 q 名最大上推
            dn = (sh[qh]   - ah) if qh < n    else -INF   # 第 q+1 名最大下压
            if not (up > tau and dn <= tau): ok = False; break
        if ok: return True
    return False


def brute_shift(S, a, q, grid=2001):
    """暴力：**纯平移**族（单个 c_h，|c_h| ≤ a_h）能否达成 q —— 验证「可达集相同」。"""
    allv = np.concatenate([np.asarray(x) for x in S])
    lo, hi = allv.min()-max(a)-1, allv.max()+max(a)+1
    for tau in np.linspace(lo, hi, grid):
        ok = True
        for sh, ah, qh in zip(S, a, q):
            n = len(sh)
            c_lo = (tau - sh[qh-1]) if qh >= 1 else -INF   # c 必须 >
            c_hi = (tau - sh[qh])   if qh < n    else  INF # c 必须 ≤
            # 与 [−a, a] 求交
            L = max(c_lo, -ah); R = min(c_hi, ah)
            if not (R > L or (np.isclose(R, L) and R <= ah and R >= -ah)):
                ok = False; break
        if ok: return True
    return False


CASES = [
    # (名称, 每头降序分数, 每头界, 目标配额, 期望可达)
    ("① 可达：界足够",        [[1.0,0.4],[0.9,0.3]], [0.5,0.5], [1,1], True),
    ("② 不可达：界太小",      [[0.0,-1.0],[1.0,0.9]], [0.05,0.05],[2,0], False),
    ("③ 边界 q=0（整头清空）", [[0.3,0.2]],           [0.5],      [0],   True),
    ("④ 边界 q=n（整头保留）", [[0.3,0.2]],           [0.5],      [2],   True),
    ("⑤ 平局：s_(q)=s_(q+1)", [[0.5,0.5,0.1]],       [0.0],      [1],   False),
    ("⑥ 平局但界>0 可绕开",   [[0.5,0.5,0.1]],       [0.3],      [1],   True),
    ("⑦ 两头竞争、恰好卡住",  [[0.0,-0.1],[0.5,0.4]],[0.05,0.05],[2,0], False),
    ("⑧ 同一对分数、界够大即可达",[[0.0,-1.0],[1.0,0.9]], [1.10,1.10],[2,0], True),
]

def main():
    bad = 0
    print(f"{'用例':<26}{'slack':>10}{'判据':>7}{'暴力':>7}{'期望':>7}  结果")
    for nm, S, a, q, exp in CASES:
        S = [np.asarray(x, float) for x in S]
        sl = slack(S, a, q); pred = sl > 0; bf = brute(S, a, q)
        ok = (pred == bf) and (bf == exp)   # 判据必须等于暴力；期望列只是自检
        bad += (not ok)
        print(f"{nm:<26}{sl:>+10.4f}{str(pred):>7}{str(bf):>7}{str(exp):>7}  "
              f"{'OK' if ok else '**FAIL**'}")
    print("\n【推论检验】有界平移与有界逐 token 的可达集是否相同（配额只取决于边界对）")
    same = 0
    rng = np.random.default_rng(0)
    for t in range(300):
        H = int(rng.integers(1,4)); S=[]; a=[]; q=[]
        for _ in range(H):
            n = int(rng.integers(2,6))
            sh = np.sort(rng.normal(size=n))[::-1].copy()
            S.append(sh); a.append(float(abs(rng.normal())*0.3)); q.append(int(rng.integers(0,n+1)))
        same += int(brute(S,a,q) == brute_shift(S,a,q))
    print(f"  300 个随机例子中两族判定一致：**{same}/300**"
          f"   {'✓ 与推导一致' if same==300 else '✗ 推导有误，需重查'}")
    bad += (same != 300)
    print(f"\n{'全部通过' if bad==0 else f'**{bad} 项 FAIL**'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
