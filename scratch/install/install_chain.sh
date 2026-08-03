#!/usr/bin/env bash
# Stage 0 dependency install chain. Runs to completion; prints STEP/OK/FAIL markers.
set -o pipefail
cd /home/ubuntu/zxy/vlm-memory
PY=.venv/bin/python

echo "STEP wait-for-torch"
# Wait for any in-flight torch pip to finish
while pgrep -f "pip install torch==2.7.0" >/dev/null 2>&1; do sleep 5; done

echo "STEP verify-torch"
if ! $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"; then
  echo "STEP install-torch (was missing)"
  $PY -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128 || { echo "FAIL torch"; exit 1; }
  $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || { echo "FAIL torch-import"; exit 1; }
fi
echo "OK torch"

echo "STEP install-flash-attn"
$PY -m pip install flash-attn==2.7.3 --no-build-isolation 2>&1 | tail -40
$PY -c "import flash_attn; print('flash_attn', flash_attn.__version__)" || { echo "FAIL flash-attn"; exit 1; }
echo "OK flash-attn"

echo "STEP install-requirements"
$PY -m pip install -r external/FastKVzip/prefill/requirements.txt 2>&1 | tail -30 || { echo "FAIL requirements"; exit 1; }
echo "OK requirements"

echo "STEP verify-imports"
$PY -c "import torch, flash_attn, transformers, datasets, accelerate; print('transformers', transformers.__version__); print('ALL_IMPORTS_OK')" || { echo "FAIL verify-imports"; exit 1; }
echo "DONE stage0-install"
