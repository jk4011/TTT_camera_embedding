#!/usr/bin/env bash
# Q33: LaCT-LVSM on gObjaverse -- the OBJECT-level end of the camera-baseline axis.
#
#   ./run_gobj_grid.sh "0 1 2 3" [seed]
#
# WHY. Q32 asks whether camera baseline is the lever that decides when the input and
# hidden rotary sites COMPOSE. This grid puts the far end of that axis on the board,
# measured with one method across all three datasets (angle between camera forward
# axes, over the views the loader actually serves):
#
#     RE10K ~7 deg   ->   DL3DV 34.5 deg   ->   gObjaverse 91.2 deg
#
# Cameras orbit the object on a sphere, so pairwise angles reach a full 180 deg. If
# `both` subtracts because wrapped ladder frequencies act as decorrelation, this is
# where the effect should be largest and easiest to see.
#
# It is also the setting LaCT itself used: its object-level checkpoint was trained on
# Objaverse renders (32 views/object, LVSM/GS-LRM settings). Those renders were never
# released; gObjaverse is the public pre-rendered stand-in, at 40 views/object.
#
# SCOPE, to be stated in any write-up: object-level moves MORE than the baseline --
# background, object-centric normalization and bounded scene scale change too. This is
# NOT a clean single-variable extension of the DL3DV grid. Read it as coverage of
# LaCT's own object-level setting plus a directional check on the geometry story,
# never as an isolated baseline experiment.
#
# Data must exist first (reshard_gobjaverse.py -> /tmp/gobj). Standard protocol
# otherwise: 30k steps, bs16, lr 1e-4, 8+8 views, 256x256, LPIPS from 5k, eval on the
# 200-scene deterministic holdout, 8 uniform inputs / 4 midpoint targets.
set -u
GPUS=${1:-"0 1 2 3"}; SEED=${2:-95}
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1

[ -f /tmp/gobj/train_index.json ] || { echo "FATAL: /tmp/gobj/train_index.json missing -- run reshard_gobjaverse.py first"; exit 1; }
[ -f /tmp/gobj/test_index.json ]  || { echo "FATAL: test index missing"; exit 1; }

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
  local EXP="gobj_${ARM}_s${SEED}"
  [ -f "outputs/$EXP/eval.json" ] && { echo "[$ARM] done"; return; }
  DATA_PATH=/tmp/gobj/train_index.json ./launch_exp.sh "$GPU" "$EXP" "${CFG[$ARM]}" "$SEED"
  CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$EXP/model_0030000.pth" \
    --config "${CFG[$ARM]}" --data_path /tmp/gobj/test_index.json --num_scenes 200 \
    > "outputs/$EXP/eval.log" 2>&1
  echo "[$ARM] eval exit=$?"
}
for i in "${!ARMS[@]}"; do run_one "${G[$i]}" "${ARMS[$i]}" & sleep 5; done
wait
echo "[gobj grid] all four arms done"
