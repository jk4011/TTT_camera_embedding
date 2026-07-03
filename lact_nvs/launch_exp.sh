#!/bin/bash
# Usage: launch_exp.sh <gpu> <expname> <config>
# Single-GPU 30k-iter training run with the standard experiment protocol.
set -u
GPU=$1
EXP=$2
CONFIG=$3

PY_ENV=/NHNHOME/WORKSPACE/26msit001_T_B/POSTECH-CGLAB/anaconda3/envs/LVSM/bin
cd "$(dirname "$0")"
mkdir -p outputs/$EXP

CUDA_VISIBLE_DEVICES=$GPU $PY_ENV/torchrun \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
  train.py \
  --config $CONFIG \
  --data_path /tmp/re10k/train_index.json --dataset re10k --scene_pose_normalize \
  --expname $EXP \
  --steps 30000 --warmup 1500 --lr 1e-4 --lpips_start 5000 \
  --bs_per_gpu 16 --num_all_views 15 --num_input_views 8 --num_target_views 8 \
  --image_size 256 256 --num_workers 7 \
  --save_every 10000 --log_every 200 \
  > outputs/$EXP/train.log 2>&1
echo "EXIT $? $EXP" >> outputs/exp_status.log
