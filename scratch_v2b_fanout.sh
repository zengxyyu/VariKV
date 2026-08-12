#!/usr/bin/env bash
# v2b dist 在 Retr.KV 上的结果一出来，就决定是否扩散到其余 10 个面板。
#
# 为什么要加门槛：v2b 是"+21.60 是真的还是运气"的判定臂。若它在 Retr.KV 上也是
# 负的（v2a 是 −13.20），那么再花 4 GPU-h 把它铺到 10 个面板上只会得到 10 个
# 负结果 —— 信息量低。所以先看 Retr.KV，过了门槛才扩散。
#
# 门槛定得很低（Δ ≥ +5）：只要方向对就扩散，不要求复现全部 21.60。
# 想无条件扩散：FORCE=1 ./scratch_v2b_fanout.sh
set -u; cd "$(dirname "$0")" || exit 1
LOG=scratch_centroid_logs/klv2b_dist.log
echo "$(date -u +%H:%M) 等 v2b dist 的 Retr.KV 评测跑完"
for i in $(seq 1 480); do grep -q "Finished." "$LOG" 2>/dev/null && break; sleep 60; done
grep -q "Finished." "$LOG" || { echo "超时，未完成"; exit 1; }

DELTA=$(.venv/bin/python - <<'PY'
import sys, os, importlib.util, numpy as np
ROOT=os.getcwd(); sys.argv=[sys.argv[0]]
spec=importlib.util.spec_from_file_location("rep",os.path.join(ROOT,"scratch_klsweep_report.py"))
rep=importlib.util.module_from_spec(spec); spec.loader.exec_module(rep)
k,_=rep.per_sample("scbench_kv","_klv2b_dist_chunk16k_w4096_varikvdist16_res")
b,_=rep.per_sample("scbench_kv","_b01_chunk16k_w4096")
c=sorted(set(k)&set(b))
print(f"{(np.mean([k[i] for i in c])-np.mean([b[i] for i in c]))*100:.2f}")
PY
)
echo "$(date -u +%H:%M) v2b dist 在 Retr.KV 上的 Δ = $DELTA"
PASS=$(.venv/bin/python -c "print(1 if ${DELTA} >= 5.0 else 0)")
if [ "${FORCE:-0}" != "1" ] && [ "$PASS" != "1" ]; then
  echo "未过门槛（Δ < +5）⇒ **不扩散**。若仍要跑：FORCE=1 ./scratch_v2b_fanout.sh"
  exit 0
fi
echo "$(date -u +%H:%M) 过门槛 ⇒ 把 10 个面板排进 p234 队列"
CK=../../../varikv/ckpt_kl_v2b/s2b_dist_k16.pt
# 便宜的先排（qa_eng .12 → repoqa 1.17），基线全部复用 _kls_base / _fig11_*_base
for ds in scbench_choice_eng squad scbench_qa_eng gsm scbench_many_shot \
          scbench_summary scbench_vt scbench_mf scbench_prefix_suffix scbench_repoqa; do
  echo "v2b11_${ds}|${ds}|--varikv_ckpt $CK --varikv_residual --varikv_slots 16" \
    >> scratch_p234_logs/.queue
done
echo "$(date -u +%H:%M) 已排入 10 个，队列现有 $(wc -l < scratch_p234_logs/.queue) 个"
