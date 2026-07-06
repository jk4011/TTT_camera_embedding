#!/bin/bash
# Q1 absolute-adaptation probe: 30k-iter training then standard 256-scene eval.
# Usage: nohup bash run_q1_chain.sh > /tmp/q1_chain.log 2>&1 &
cd "$(dirname "$0")"
PY=/NHNHOME/WORKSPACE/26msit001_T_B/POSTECH-CGLAB/anaconda3/envs/LVSM/bin/python
GPU=3
EXP=q1_scenerand
CFG=config/cam_q1_scenerand.yaml
SEED=95

echo "[q1_chain] $(date) training start"
# robust compile env: fresh NFS caches (exec-allowed), no async compile pool
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

bash launch_exp.sh $GPU $EXP $CFG $SEED
echo "[q1_chain] $(date) training done"

CKPT=outputs/$EXP/model_0030000.pth
if [ ! -f "$CKPT" ]; then
  echo "[q1_chain] ERROR: $CKPT missing, skipping eval (see outputs/$EXP/train.log)"
  exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load $CKPT --config $CFG \
  > outputs/$EXP/eval.log 2>&1
echo "[q1_chain] $(date) eval done:"
tail -2 outputs/$EXP/eval.log
