#!/bin/bash
# Q2 depth-3 fast weights ("one rotary per address space"): three sequential
# 30k-iter trainings (fw3l / fw3l_rot2 / fw3l_rot3) + standard 256-scene evals.
# Usage: nohup bash run_q2_chain.sh > /tmp/q2_chain.log 2>&1 &
cd "$(dirname "$0")"
PY=/NHNHOME/WORKSPACE/26msit001_T_B/POSTECH-CGLAB/anaconda3/envs/LVSM/bin/python
GPU=3
SEED=95

# robust compile env: fresh NFS caches (exec-allowed), no async compile pool
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

for EXP in fw3l_base fw3l_rot2 fw3l_rot3; do
  CFG=config/cam_$EXP.yaml
  echo "[q2_chain] $(date) training start: $EXP"
  bash launch_exp.sh $GPU $EXP $CFG $SEED
  echo "[q2_chain] $(date) training done: $EXP"

  CKPT=outputs/$EXP/model_0030000.pth
  if [ ! -f "$CKPT" ]; then
    echo "[q2_chain] ERROR: $CKPT missing, skipping eval (see outputs/$EXP/train.log)"
    continue
  fi

  CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load $CKPT --config $CFG \
    > outputs/$EXP/eval.log 2>&1
  echo "[q2_chain] $(date) eval done: $EXP"
  tail -2 outputs/$EXP/eval.log
done
echo "[q2_chain] $(date) ALL DONE"
