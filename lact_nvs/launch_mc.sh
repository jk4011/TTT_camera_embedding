#!/bin/bash
# Multi-chunk training runs (NODE2_PROMPT_MULTICHUNK.md): standard protocol with the
# view count raised to 32 inputs, and ttt_num_chunks: [1,2,4,8] set in the config.
# Usage: launch_mc.sh <gpu> <expname> <config> [seed]
#
# Differences from launch_exp.sh, and only these:
#   --num_input_views  8 -> 32   (32 views = 8192 update tokens -> chunks 8192/4096/2048/1024)
#   --num_all_views   15 -> 39   (same input/target overlap pattern as the 8+8 standard,
#                                 which draws inputs [:8] and targets [-8:] out of 15)
# Everything else — 30k iters, bs16, lr 1e-4, warmup 1500, lpips from 5k, seed 95,
# 256x256, 8 target views — is the standard protocol.
#
# nproc_per_node stays 1: model.py draws one n per forward, which is only safe on a
# single rank (each rank would otherwise build a different graph).
set -u
GPU=$1
EXP=$2
CONFIG=$3
SEED=${4:-95}

PY_ENV=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin
cd "$(dirname "$0")"
mkdir -p outputs/$EXP

REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

CUDA_VISIBLE_DEVICES=$GPU $PY_ENV/torchrun \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
  train.py \
  --config $CONFIG \
  --data_path /tmp/re10k/train_index.json --dataset re10k --scene_pose_normalize \
  --expname $EXP \
  --steps 30000 --warmup 1500 --lr 1e-4 --lpips_start 5000 --seed $SEED \
  --bs_per_gpu 16 --num_all_views 39 --num_input_views 32 --num_target_views 8 \
  --image_size 256 256 --num_workers 7 \
  --save_every 10000 --log_every 200 \
  > outputs/$EXP/train.log 2>&1
echo "EXIT $? $EXP" >> outputs/exp_status.log
