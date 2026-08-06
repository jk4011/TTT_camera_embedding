#!/usr/bin/env bash
# Q38: orthogonal group action (user design, 2026-08-07) at the TTT input site.
#   ./run_ogta.sh <gpu> <data: gobj|re10k>
# Per-view block-diagonal ORTHOGONAL matrix on fast q/k after the L2-norm:
# exact c2w rotation (3x3) + one SO(2) per translation axis, ladder capped at pi/2
# so translation phases NEVER wrap (|t|<=1 after scene normalisation). 28 units =
# 252/256 dims. Rationale: Muon + weight-norm + q/k L2-norm all live on spheres;
# an orthogonal address transform composes with them without norm distortion, and
# the wrap-prone part (rotation at wide baselines) enters exactly.
set -u
GPU=$1; DATA=$2
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
if [ "$DATA" = gobj ]; then
  TRAIN=/tmp/gobj/train_index.json; TEST=/tmp/gobj/test_index.json
  EXTRA="--min_frames 40"; NSC=500; EXP=gobj_ogta_s95
else
  TRAIN=/tmp/re10k/train_index.json; TEST=/tmp/re10k/test_index.json
  EXTRA=""; NSC=256; EXP=ogta_s95
fi
[ -f "$TRAIN" ] || { echo "FATAL: $TRAIN missing"; exit 1; }
[ -f "outputs/$EXP/eval.json" ] && { echo "[$EXP] done"; exit 0; }
mkdir -p "outputs/$EXP"
CUDA_VISIBLE_DEVICES=$GPU $PY -m torch.distributed.run \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
  train.py --config config/cam_ogta.yaml \
  --data_path "$TRAIN" --dataset re10k --scene_pose_normalize $EXTRA \
  --expname "$EXP" \
  --steps 30000 --warmup 1500 --lr 1e-4 --lpips_start 5000 --seed 95 \
  --bs_per_gpu 16 --num_all_views 15 --num_input_views 8 --num_target_views 8 \
  --image_size 256 256 --num_workers 7 \
  --save_every 10000 --log_every 200 \
  > "outputs/$EXP/train.log" 2>&1
echo "EXIT $? $EXP" >> outputs/exp_status.log
CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$EXP/model_0030000.pth" \
  --config config/cam_ogta.yaml --data_path "$TEST" --num_scenes $NSC \
  > "outputs/$EXP/eval.log" 2>&1
echo "[$EXP] eval exit=$?"
