#!/bin/bash
# Budget-sensitivity run: is the rotary's advantage an artefact of the 30k budget?
#
#   bash run_budget80k.sh <gpu> <arm>          # one arm on one GPU
#   GPUS="0 1 2 3" bash run_budget80k.sh       # all four, one per GPU
#
# WHY. The 30k protocol is a fixed budget, not a convergence criterion: train.py's own
# default is 80000, and base_s95's train PSNR is still climbing in its last window
# (20.855 -> 21.061 over the final 5k steps). Two things follow that a reviewer will
# ask about, and we currently cannot answer:
#   1. Does the rotary's gain over NoPE survive a longer budget, or does NoPE catch up?
#   2. Both's lead over hidden-only DECAYS monotonically through training
#      (+0.398 at 2.5k -> +0.056 at 15k -> -0.001 at 27.5k, train PSNR). Extrapolated,
#      hidden-only could OVERTAKE Both at 80k. This is why hidden is not optional here.
#
# Same protocol as the 30k runs in every other respect: fixed ladders (input F21 /
# hidden F_h42), seed 95, bs16, lr 1e-4, 8+8 views, 256x256, LPIPS from step 5000.
# warmup scales 1500 -> 4000 so the cosine keeps its shape over the longer horizon.
#
# Cost: ~4.3 h per arm on one B200, four arms in parallel on four GPUs.
# These numbers are comparable ONLY to each other, never to the 30k table.
set -u
cd "$(dirname "$0")"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python

declare -A CFG=(
  [base]=config/lact_l6_d256_p16.yaml
  [input]=config/cam_pra_hi.yaml
  [hidden]=config/cam_h_pra_hi.yaml
  [both]=config/cam_pra_h_hi.yaml
)

run_one() {
  local GPU=$1 ARM=$2
  local EXP="b80k_${ARM}_s95"
  local C=${CFG[$ARM]}
  if [ -f "outputs/$EXP/eval.json" ]; then echo "[$ARM] already done"; return; fi
  echo "[$ARM] GPU $GPU -> outputs/$EXP"
  STEPS=80000 WARMUP=4000 ./launch_exp.sh "$GPU" "$EXP" "$C" 95
  # eval the 80k checkpoint, not whatever save_every last wrote
  CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$EXP/model_0080000.pth" \
    --config "$C" > "outputs/$EXP/eval.log" 2>&1
  echo "[$ARM] eval exit=$?"
}

if [ $# -ge 2 ]; then
  run_one "$1" "$2"
else
  read -r -a G <<< "${GPUS:-0 1 2 3}"
  ARMS=(base input hidden both)
  for i in "${!ARMS[@]}"; do run_one "${G[$i]}" "${ARMS[$i]}" & sleep 5; done
  wait
fi
echo "[b80k] done"
