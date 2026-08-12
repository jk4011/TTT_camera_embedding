#!/bin/bash
# Retrain the NoPE baseline at seeds 137/211 (checkpoints lost in the 2026-07-09
# reset; Table 1's 3-seed NoPE row needs SSIM, which requires re-evaluating a
# checkpoint). Standard 30k protocol via launch_exp.sh, then SSIM eval.
# One seed per GPU, co-residing with f85.
set -u
cd "$(dirname "$0")"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python

run_seed() {
  local gpu=$1 seed=$2
  bash launch_exp.sh "$gpu" "base_s${seed}" config/lact_l6_d256_p16.yaml "$seed"
  CUDA_VISIBLE_DEVICES=$gpu $PY eval.py \
    --load "outputs/base_s${seed}/model_0030000.pth" \
    --config config/lact_l6_d256_p16.yaml \
    --out "outputs/base_s${seed}/eval_ssim.json" \
    > "outputs/base_s${seed}/eval_ssim.log" 2>&1
  echo "[base_s${seed}] eval exit=$?"
}
run_seed 2 137 &
run_seed 3 211 &
wait
echo "BASE SEED RETRAINS DONE $(date)"
