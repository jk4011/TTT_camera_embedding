#!/usr/bin/env bash
# Q34: LVSM on DL3DV at 32 INPUT VIEWS -- does view density restore composition?
#
#   ./run_dl3dv_v32_grid.sh "0 1 2 3" [seed]
#
# THE HYPOTHESIS (user, 2026-08-06). Raising the view count in the SAME dataset packs
# the window with overlapping poses, so camera addressing should matter more, and the
# DL3DV composition failure (F50: both - best single = -0.148) might reverse.
#
# WHAT VIEW COUNT ACTUALLY CHANGES, measured through the eval loader on this data:
#
#   input views          8      16      32
#   median PAIRWISE    45.3    40.6    38.7   deg   <- set by the window, ~unchanged
#   nearest-NEIGHBOUR  12.8     8.0     4.7   deg   <- RE10K-like at 32 views
#
# So this is a clean discriminator between two accounts of the failure:
#   - if retrieval leans on NEAREST-neighbour pairs, 32 views is locally RE10K
#     (~5-7 deg) and `both` should recover;
#   - if the TYPICAL pair is what matters, the median stays ~39 deg and the failure
#     stays. F48 (tttLRM, eval-only extrapolation) points this way -- both got
#     monotonically WORSE toward 32 views -- but phases there were trained at 8
#     views, so it does not answer the training question. This grid does.
#
# Protocol: 32 input + 8 target (num_all_views 40), otherwise the F50 recipe. Same
# 140-scene eval split, 32 uniform inputs / 4 midpoint targets. ~2.7x the tokens of
# the 8-view grid per step; expect ~5-6 h per arm.
set -u
GPUS=${1:-"0 1 2 3"}; SEED=${2:-95}
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
[ -f /tmp/dl3dv/train_index.json ] || { echo "FATAL: reshard first"; exit 1; }

declare -A CFG=(
  [base]=config/lact_l6_d256_p16.yaml
  [input]=config/cam_pra_hi.yaml
  [hidden]=config/cam_h_pra_hi.yaml
  [both]=config/cam_pra_h_hi.yaml
)
ARMS=(base input hidden both)
read -r -a G <<< "$GPUS"

run_one() {
  local GPU=$1 ARM=$2
  local EXP="dl3dv32_${ARM}_s${SEED}"
  [ -f "outputs/$EXP/eval.json" ] && { echo "[$ARM] done"; return; }
  mkdir -p "outputs/$EXP"
  # bs halved (16 -> 8): 40 views is 2.7x the tokens of the standard 15, and bs16
  # does not fit. Same for all arms, so the contrast is unharmed.
  CUDA_VISIBLE_DEVICES=$GPU $PY -m torch.distributed.run \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
    train.py --config "${CFG[$ARM]}" \
    --data_path /tmp/dl3dv/train_index.json --dataset re10k --scene_pose_normalize \
    --expname "$EXP" \
    --steps 30000 --warmup 1500 --lr 1e-4 --lpips_start 5000 --seed "$SEED" \
    --bs_per_gpu 8 --num_all_views 40 --num_input_views 32 --num_target_views 8 \
    --image_size 256 256 --num_workers 7 \
    --save_every 10000 --log_every 200 \
    > "outputs/$EXP/train.log" 2>&1
  echo "EXIT $? $EXP" >> outputs/exp_status.log
  CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$EXP/model_0030000.pth" \
    --config "${CFG[$ARM]}" --data_path /tmp/dl3dv/test_index.json \
    --num_scenes 140 --num_input_views 32 \
    > "outputs/$EXP/eval.log" 2>&1
  echo "[$ARM] eval exit=$?"
}
if [ "${#G[@]}" -eq 1 ]; then
  # single GPU given -> run the four arms SEQUENTIALLY on it (node2 mode, like the
  # mc grid). Order: base first so the reference exists early, both last.
  for ARM in base input hidden both; do run_one "${G[0]}" "$ARM"; done
else
  for i in "${!ARMS[@]}"; do run_one "${G[$i]}" "${ARMS[$i]}" & sleep 5; done
  wait
fi
echo "[dl3dv32] all arms done"
