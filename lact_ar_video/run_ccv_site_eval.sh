#!/usr/bin/env bash
# P5 phase-1 eval: paired held-out val loss for the three site-ablation cells, at the
# SAME common checkpoint step F30 used (13999), so the numbers line up with the
# existing ccv_base / ccv_pra / ccv_pra_fixed / ccv_both rows.
# Usage: ./run_ccv_site_eval.sh <cell: in|h|both> <gpu> [step]
#
# Uses the archived copy of step 13999 (outputs/_keep_step13999/) when present: the
# live checkpoint dir is subject to keep_last_iter pruning.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CELL="$1"; GPU="$2"; STEP="${3:-013999}"
EXP="ccv_site_${CELL}"
CFG="configs/ar/abl_ccv_site_${CELL}.yaml"

ARCHIVED="$PWD/outputs/_keep_step13999/${EXP}_checkpoint_model_${STEP}"
LIVE="$PWD/outputs/${EXP}/seed_1/checkpoint_model_${STEP}"
if [ -d "$ARCHIVED" ]; then CKPT="$ARCHIVED"; else CKPT="$LIVE"; fi
[ -d "$CKPT" ] || { echo "FATAL: no checkpoint at step $STEP for $EXP"; exit 1; }

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
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/eval_site

cd minVid
echo "[eval] $EXP @ step $STEP  ckpt=$CKPT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" eval_ccv_valloss.py \
    --config "$CFG" --ckpt "$CKPT" \
    --out "../outputs/eval_site/valloss_${EXP}_${STEP}.json" \
    > "../outputs/eval_site/valloss_${EXP}_${STEP}.log" 2>&1
echo "[eval] $EXP exit=$?"
