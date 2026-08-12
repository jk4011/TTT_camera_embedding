#!/usr/bin/env bash
# Q36: is gObjaverse's rotary failure ALIASING or the site itself? One GPU, two cells
# sequentially: the faithful PRoPE port (exact projective group action, NO frequency
# ladder on camera coords) and GTA (exact SE(3) action), both at the TTT site, on
# gObjaverse -- the dataset where every sinusoidal arm lost (F51).
#
#   ./run_gobj_groupaction.sh <gpu>
#
# WHY. The PRoPE paper's Table 1 shows PRoPE/GTA at +2.3 dB over the Plucker-raymap
# baseline ON OBJAVERSE, trained at 2 context views on an object orbit -- a WIDE
# baseline, so the narrow-geometry reconciliation does not apply there. Their encoding
# is an exact relative transform P_j^-1 P_i: no sinusoid over camera coordinates, so
# NOTHING WRAPS at any baseline. Our F51 arms are sinusoidal Plucker ladders, which at
# ~91 deg alias almost everywhere. Two load-bearing differences remain: encoding
# structure (group action vs ladder) and site (attention vs fast weights).
#
#   prope_orig/gta >= base on gObjaverse  -> aliasing is the culprit; wrap-free
#       addressing at the TTT site survives wide baselines, and becomes the design
#       direction for the wide-baseline regime.
#   they lose too -> the TTT site itself cannot exploit relative pose at extreme
#       baselines, whatever the encoding -- the site distinction carries everything.
#
# Note F34 context: at the TTT site on RE10K, the projective action alone LOST
# (prope_raw -0.294) and GTA was flat (-0.02, 3-seed). So a win here would be
# surprising and decisive; a loss extends F34's pattern to the wide-baseline end.
set -u
[ $# -ge 1 ] || { echo "usage: $0 <gpu>" >&2; exit 1; }
GPU=$1
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
[ -f /tmp/gobj/train_index.json ] || { echo "FATAL: reshard gobj first"; exit 1; }

run_one() {
  local EXP=$1 CFG=$2
  [ -f "outputs/$EXP/eval.json" ] && { echo "[$EXP] done"; return; }
  mkdir -p "outputs/$EXP"
  CUDA_VISIBLE_DEVICES=$GPU $PY -m torch.distributed.run \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
    train.py --config "$CFG" \
    --data_path /tmp/gobj/train_index.json --dataset re10k --scene_pose_normalize \
    --min_frames 40 \
    --expname "$EXP" \
    --steps 30000 --warmup 1500 --lr 1e-4 --lpips_start 5000 --seed 95 \
    --bs_per_gpu 16 --num_all_views 15 --num_input_views 8 --num_target_views 8 \
    --image_size 256 256 --num_workers 7 \
    --save_every 10000 --log_every 200 \
    > "outputs/$EXP/train.log" 2>&1
  echo "EXIT $? $EXP" >> outputs/exp_status.log
  CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$EXP/model_0030000.pth" \
    --config "$CFG" --data_path /tmp/gobj/test_index.json --num_scenes 500 \
    > "outputs/$EXP/eval.log" 2>&1
  echo "[$EXP] eval exit=$?"
}
run_one gobj_prope_orig_s95 config/cam_prope_orig.yaml
run_one gobj_gta_in_s95     config/cam_gta_in.yaml
echo "[q36] both cells done"
