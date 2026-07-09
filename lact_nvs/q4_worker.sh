#!/bin/bash
# q4_worker.sh <gpu> <job1> <job2> ...   job = "<variant>:<seed>"
# Runs each job (30k train via launch_exp.sh, then 256-scene eval) sequentially
# on one GPU. Full Q4 3-seed ablation; all prior checkpoints were lost to the
# host reset so every (variant, seed) is retrained from scratch.
set -u
cd "$(dirname "$0")"
GPU=$1; shift
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
export TRITON_CACHE_DIR="$PWD/../.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$PWD/../.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
for job in "$@"; do
  V="${job%%:*}"; S="${job##*:}"; EXP="${V}_s${S}"; CFG="config/cam_${V}.yaml"
  echo "[q4 gpu$GPU] $(date '+%F %T') START $EXP"
  bash launch_exp.sh "$GPU" "$EXP" "$CFG" "$S"
  CKPT="outputs/$EXP/model_0030000.pth"
  if [ -f "$CKPT" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "$CKPT" --config "$CFG" \
      > "outputs/$EXP/eval.log" 2>&1
    echo "[q4 gpu$GPU] $(date '+%F %T') EVAL $EXP: $(grep -h PSNR outputs/$EXP/eval.log | tail -1)"
  else
    echo "[q4 gpu$GPU] ERROR $EXP: checkpoint missing (train failed); see outputs/$EXP/train.log"
  fi
done
echo "[q4 gpu$GPU] $(date '+%F %T') WORKER DONE"
