#!/bin/bash
# Usage: launch_exp.sh <gpu> <expname> <config>
# Single-GPU 30k-iter training run with the standard experiment protocol.
set -u
GPU=$1
EXP=$2
CONFIG=$3
SEED=${4:-95}
# Budget overrides. Defaults reproduce the standard 30k protocol exactly, so every
# existing queue entry is untouched. STEPS=80000 WARMUP=4000 is the budget-sensitivity
# run: warmup scales with the budget (1500 x 80/30 = 4000, which is also train.py's own
# default at 80k) so the cosine keeps its shape. lpips_start stays ABSOLUTE at 5000,
# which preserves the early-training recipe rather than the LPIPS-on fraction.
STEPS=${STEPS:-30000}
WARMUP=${WARMUP:-1500}
# Dataset overrides for the DL3DV cross-data run. Defaults reproduce RE10K exactly.
# DL3DV is resharded into the RE10K per-scene .torch format (reshard_dl3dv.py), so
# DATASET stays "re10k" even for DL3DV -- it names the FORMAT, not the corpus.
DATA_PATH=${DATA_PATH:-/tmp/re10k/train_index.json}
DATASET=${DATASET:-re10k}

PY_ENV=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin
cd "$(dirname "$0")"
mkdir -p outputs/$EXP

# Dedicated NFS compile caches: default /tmp/torchinductor_* is noexec tmpfs and
# crashes triton mid-train ("failed to map segment"); see commit a73dcd4.
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

CUDA_VISIBLE_DEVICES=$GPU $PY_ENV/torchrun \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
  train.py \
  --config $CONFIG \
  --data_path "$DATA_PATH" --dataset "$DATASET" --scene_pose_normalize \
  --expname $EXP \
  --steps $STEPS --warmup $WARMUP --lr 1e-4 --lpips_start 5000 --seed $SEED \
  --bs_per_gpu 16 --num_all_views 15 --num_input_views 8 --num_target_views 8 \
  --image_size 256 256 --num_workers 7 \
  --save_every 10000 --log_every 200 \
  > outputs/$EXP/train.log 2>&1
echo "EXIT $? $EXP" >> outputs/exp_status.log
