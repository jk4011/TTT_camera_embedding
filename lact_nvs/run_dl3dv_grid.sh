#!/usr/bin/env bash
# Q32: LaCT-LVSM on DL3DV -- the data-geometry test. Four arms, one per GPU.
#
#   ./run_dl3dv_grid.sh "0 1 2 3" [seed]
#
# THE QUESTION. On RE10K (7 deg median between-view angle) the hidden site carries most
# of the gain and both > either single site. On tttLRM/DL3DV (61 deg) both LOSES a third
# of the single-site gain (F45) and turns harmful off the trained view count (F48). But
# tttLRM also differs in backbone (24 TTT layers vs 6), head (Gaussians + depth loss vs
# RGB), and ladder. This grid changes ONLY the data: same LaCT-LVSM, same configs, same
# protocol, DL3DV instead of RE10K.
#
#   both composes on DL3DV here      -> geometry is NOT the cause; suspects: backbone
#                                       depth / output head (tttLRM-side).
#   both fails to compose here too   -> data geometry is the lever, and "when do the
#                                       sites compose" becomes a claim about CAMERA
#                                       BASELINE, testable and paper-worthy.
#
# MEASUREMENT behind it: at these datasets' cross-view |dc|, ~50% of the DL3DV ladder
# wraps (|dtheta| > pi between views) vs ~38% on RE10K; wrapped frequencies act as
# decorrelation, and a second site doubles the dose without adding information.
#
# Data must exist first (reshard_dl3dv.py -> /tmp/dl3dv). Standard protocol otherwise:
# 30k steps, bs16, lr 1e-4, 8+8 views, 256x256, LPIPS from 5k, eval on the SAME
# 140-scene test split tttLRM's F45 used, 8 uniform inputs / 4 midpoint targets.
set -u
GPUS=${1:-"0 1 2 3"}; SEED=${2:-95}
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1

[ -f /tmp/dl3dv/train_index.json ] || { echo "FATAL: /tmp/dl3dv/train_index.json missing -- run reshard_dl3dv.py first"; exit 1; }
[ -f /tmp/dl3dv/test_index.json ]  || { echo "FATAL: test index missing"; exit 1; }

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
  local EXP="dl3dv_${ARM}_s${SEED}"
  [ -f "outputs/$EXP/eval.json" ] && { echo "[$ARM] done"; return; }
  DATA_PATH=/tmp/dl3dv/train_index.json ./launch_exp.sh "$GPU" "$EXP" "${CFG[$ARM]}" "$SEED"
  CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$EXP/model_0030000.pth" \
    --config "${CFG[$ARM]}" --data_path /tmp/dl3dv/test_index.json --num_scenes 140 \
    > "outputs/$EXP/eval.log" 2>&1
  echo "[$ARM] eval exit=$?"
}
for i in "${!ARMS[@]}"; do run_one "${G[$i]}" "${ARMS[$i]}" & sleep 5; done
wait
echo "[dl3dv grid] all four arms done"
