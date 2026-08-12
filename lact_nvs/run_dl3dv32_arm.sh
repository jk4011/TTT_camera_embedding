#!/bin/bash
# Train ONE dl3dv32 baseline arm (Table-1 right columns) + SSIM eval.
# Usage: run_dl3dv32_arm.sh <gpu> <exp> <config>   e.g. 5 dl3dv32_gta_s95 config/cam_gta_in.yaml
# Same recipe as run_dl3dv_v32_grid.sh (32 in + 8 target, bs 8, 30k, seed 95).
# train.py resumes from the outputs dir, so rerunning after a reset continues.
set -u
GPU=$1; EXP=$2; CFG=$3
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
until [ -f /tmp/dl3dv/train_index.json ]; do sleep 120; done
until [ -f /tmp/dl3dv/test_index.json ]; do sleep 120; done
echo "$EXP" > outputs/.gpu_locks/$(hostname -s)_gpu${GPU}
mkdir -p "outputs/$EXP"
CUDA_VISIBLE_DEVICES=$GPU $PY -m torch.distributed.run \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
  train.py --config "$CFG" \
  --data_path /tmp/dl3dv/train_index.json --dataset re10k --scene_pose_normalize \
  --expname "$EXP" \
  --steps 30000 --warmup 1500 --lr 1e-4 --lpips_start 5000 --seed 95 \
  --bs_per_gpu 8 --num_all_views 40 --num_input_views 32 --num_target_views 8 \
  --image_size 256 256 --num_workers 7 \
  --save_every 10000 --log_every 200 \
  > "outputs/$EXP/train.log" 2>&1
echo "[$EXP] train exit=$?"
CUDA_VISIBLE_DEVICES=$GPU $PY eval.py \
  --load "outputs/$EXP/model_0030000.pth" --config "$CFG" \
  --data_path /tmp/dl3dv/test_index.json --num_scenes 140 --num_input_views 32 \
  --out "outputs/$EXP/eval_ssim.json" > "outputs/$EXP/eval_ssim.log" 2>&1
echo "[$EXP] eval exit=$?"
rm -f outputs/.gpu_locks/$(hostname -s)_gpu${GPU}
