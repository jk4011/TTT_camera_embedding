#!/bin/bash
# 3-seed replication for fw3l_rot3 and fw3l_rot2 (seeds 137, 211; seed 95 done).
cd "$(dirname "$0")"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
PYENV=/NHNHOME/WORKSPACE/26msit001_T_B/POSTECH-CGLAB/anaconda3/envs/LVSM/bin
for job in "fw3l_rot3 137" "fw3l_rot3 211" "fw3l_rot2 137" "fw3l_rot2 211"; do
  set -- $job
  V=$1; S=$2; EXP="${V}_s${S}"
  echo "[q2_seeds] $(date) start $EXP"
  bash launch_exp.sh 3 "$EXP" "config/cam_${V}.yaml" "$S"
  CKPT="outputs/$EXP/model_0030000.pth"
  if [ -f "$CKPT" ]; then
    CUDA_VISIBLE_DEVICES=3 $PYENV/python eval.py --load "$CKPT" --config "config/cam_${V}.yaml" \
      > "outputs/$EXP/eval.log" 2>&1
    echo "[q2_seeds] $(date) eval done: $EXP"; tail -2 "outputs/$EXP/eval.log"
  else
    echo "[q2_seeds] ERROR: $CKPT missing"
  fi
done
echo "[q2_seeds] ALL DONE"
