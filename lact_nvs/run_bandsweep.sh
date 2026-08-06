#!/usr/bin/env bash
# Q39b: frequency-BAND dose-response, frozen frequencies (user, 2026-08-07).
#   ./run_bandsweep.sh <gpu>     # worker; pulls cells from the claims pool
#
# The sweep brackets the wrap boundary on each dataset. omega_scale s means band
# [s*pi/2, s*16pi]; the top rung wraps once s*16pi*|dc| > pi.
#
#   gObjaverse (|dc| ~ 2):  s=1/128 [pi/256, pi/8]   deep inside wrap-free
#                           s=1/32  [pi/64, pi/2]    exactly at the boundary (Q39)
#                           s=1/8   [pi/16, 2pi]     top ~2 octaves wrapped
#                           s=1/2   [pi/4, 8pi]      most of the band wrapped
#                           s=1     (= F51 arm)      -0.41, the failure this explains
#   RE10K (|dc| ~ 0.24):    s=1/32 (Q39), 1/8, 1/2, and s=1 = pra_hi +0.51.
#
# If the gobj curve peaks at the boundary and falls on both sides, band placement
# relative to the coordinate scale is THE variable, and a scale-aware init becomes
# the single recipe for both datasets.
set -u
GPU=$1
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
CLAIM=outputs/.bandsweep_claims; mkdir -p "$CLAIM"
HOST="$(hostname -s)"
# cell = "<exp> <config> <data>"
CELLS=(
  "gobj_gentle_s128_s95  config/cam_gentle_s128.yaml gobj"
  "gobj_gentle_s8_s95    config/cam_gentle_s8.yaml   gobj"
  "gobj_gentle_s2_s95    config/cam_gentle_s2.yaml   gobj"
  "gentle_s8_s95         config/cam_gentle_s8.yaml   re10k"
  "gentle_s2_s95         config/cam_gentle_s2.yaml   re10k"
)
for row in "${CELLS[@]}"; do
  read -r EXP CFG DATA <<< "$row"
  [ -f "outputs/$EXP/eval.json" ] && continue
  if ! ( set -o noclobber; echo "$HOST gpu$GPU $$" > "$CLAIM/$EXP" ) 2>/dev/null; then continue; fi
  if [ "$DATA" = gobj ]; then TRAIN=/tmp/gobj/train_index.json; TEST=/tmp/gobj/test_index.json; X="--min_frames 40"; N=500
  else TRAIN=/tmp/re10k/train_index.json; TEST=/tmp/re10k/test_index.json; X=""; N=256; fi
  echo "[bandsweep] gpu$GPU: $EXP"
  mkdir -p "outputs/$EXP"
  CUDA_VISIBLE_DEVICES=$GPU $PY -m torch.distributed.run \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
    train.py --config "$CFG" \
    --data_path "$TRAIN" --dataset re10k --scene_pose_normalize $X \
    --expname "$EXP" \
    --steps 30000 --warmup 1500 --lr 1e-4 --lpips_start 5000 --seed 95 \
    --bs_per_gpu 16 --num_all_views 15 --num_input_views 8 --num_target_views 8 \
    --image_size 256 256 --num_workers 7 --save_every 10000 --log_every 200 \
    > "outputs/$EXP/train.log" 2>&1
  CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$EXP/model_0030000.pth" \
    --config "$CFG" --data_path "$TEST" --num_scenes $N \
    > "outputs/$EXP/eval.log" 2>&1
  echo "[bandsweep] gpu$GPU: $EXP eval exit=$?"
  [ -f "outputs/$EXP/eval.json" ] || rm -f "$CLAIM/$EXP"
done
echo "[bandsweep] gpu$GPU: pool drained"
