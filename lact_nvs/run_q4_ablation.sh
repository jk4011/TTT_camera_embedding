#!/bin/bash
# Q4 rigorous ablation at the best fixed-ladder setting (input F21 + hidden F42),
# no learnable frequencies. 3 seeds each; seed-95 runs for pra_h_hi and pra_hi exist.
cd "$(dirname "$0")"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
for job in "h_pra_hi 95" "h_pra_hi 137" "h_pra_hi 211" "pra_h_hi 137" "pra_h_hi 211" "pra_hi 137" "pra_hi 211"; do
  set -- $job
  V=$1; S=$2; EXP="${V}_s${S}"; CFG="config/cam_${V}.yaml"
  echo "[q4] $(date) start $EXP"
  bash launch_exp.sh 3 "$EXP" "$CFG" "$S"
  CKPT="outputs/$EXP/model_0030000.pth"
  if [ -f "$CKPT" ]; then
    CUDA_VISIBLE_DEVICES=3 $PY eval.py --load "$CKPT" --config "$CFG" > "outputs/$EXP/eval.log" 2>&1
    echo "[q4] $(date) eval done: $EXP"; tail -2 "outputs/$EXP/eval.log"
  else
    echo "[q4] ERROR: $CKPT missing"
  fi
done
echo "[q4] ALL DONE"
