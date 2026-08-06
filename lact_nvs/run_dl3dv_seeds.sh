#!/usr/bin/env bash
# Seed completion for F50 + the eval-only view sweep: seeds 137/211 for all four
# DL3DV 8-view arms, then re-evaluate each at v=32. The 8-view failure (-0.148) and
# the 32-view recovery (+0.092) are both single-seed today; the paper revision will
# lean on both numbers, so both get the house 3-seed treatment.
#   ./run_dl3dv_seeds.sh <gpu> <arm>   # runs s137 then s211 for that arm, plus evals
set -u
GPU=$1; ARM=$2
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
declare -A CFG=( [base]=config/lact_l6_d256_p16.yaml [input]=config/cam_pra_hi.yaml
                 [hidden]=config/cam_h_pra_hi.yaml [both]=config/cam_pra_h_hi.yaml )
for SEED in 137 211; do
  EXP="dl3dv_${ARM}_s${SEED}"
  if [ ! -f "outputs/$EXP/eval.json" ]; then
    DATA_PATH=/tmp/dl3dv/train_index.json ./launch_exp.sh "$GPU" "$EXP" "${CFG[$ARM]}" "$SEED"
    CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$EXP/model_0030000.pth" \
      --config "${CFG[$ARM]}" --data_path /tmp/dl3dv/test_index.json --num_scenes 140 \
      > "outputs/$EXP/eval.log" 2>&1
  fi
  if [ ! -f "outputs/$EXP/eval_v32.json" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$EXP/model_0030000.pth" \
      --config "${CFG[$ARM]}" --data_path /tmp/dl3dv/test_index.json --num_scenes 140 \
      --num_input_views 32 --out "outputs/$EXP/eval_v32.json" \
      > "outputs/$EXP/eval_v32.log" 2>&1
  fi
done
echo "[$ARM] both seeds done"
