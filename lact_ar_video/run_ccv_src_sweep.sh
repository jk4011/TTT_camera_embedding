#!/bin/bash
# Fig.2 CCV panel: val loss vs SOURCE length (1/3/5/7 AR chunks) for the four F54
# cells. Same 64-pair deterministic protocol as the val-loss table; c=7 must
# reproduce the F54 numbers (built-in sanity check).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=${GPU:-4}
REPO_ROOT="$(cd .. && pwd)"
PY="$REPO_ROOT/.venv_llm/bin/python"
export HF_HOME=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/hf_cache
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
export C_INCLUDE_PATH=/usr/local/cuda/include
export PATH="/usr/local/cuda/bin:$(dirname "$PY"):$PATH"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor"
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH="$PWD"
AV_ROOT="$PWD"
ckpt_for() {
  local EXP=$1
  local A="$AV_ROOT/outputs/_keep_step13999/${EXP}_checkpoint_model_013999"
  local L="$AV_ROOT/outputs/${EXP}/seed_1/checkpoint_model_013999"
  if [ -d "$A" ]; then echo "$A"; else echo "$L"; fi
}
declare -A CFG=( [base]=configs/ar/abl_ccv_base.yaml [in]=configs/ar/abl_ccv_site_in.yaml
                 [h]=configs/ar/abl_ccv_site_h.yaml [both]=configs/ar/abl_ccv_site_both.yaml )
declare -A EXP=( [base]=ccv_base [in]=ccv_site_in [h]=ccv_site_h [both]=ccv_site_both )
mkdir -p outputs/eval_site
for C in 7 5 3 1; do
  for cell in base in h both; do
    OUT="$AV_ROOT/outputs/eval_site/valloss_srcsweep_${cell}_c${C}.json"
    [ -f "$OUT" ] && continue
    ( cd minVid && CUDA_VISIBLE_DEVICES=$GPU $PY eval_ccv_valloss.py \
        --config "${CFG[$cell]}" --ckpt "$(ckpt_for "${EXP[$cell]}")" \
        --src_chunks $C --out "$OUT" \
        > "$AV_ROOT/outputs/eval_site/srcsweep_${cell}_c${C}.log" 2>&1 )
    echo "[c$C $cell] exit=$?"
  done
done
echo "CCV SRC SWEEP DONE $(date)"
