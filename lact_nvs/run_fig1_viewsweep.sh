#!/bin/bash
# Figure 1, NVS panel: PSNR vs number of input views, one curve per rotary arm.
#
#   bash run_fig1_viewsweep.sh 0            # all arms, all view counts, on GPU 0
#   VIEWS="4 8 16" ARMS="pra_h_hi" bash run_fig1_viewsweep.sh 3
#
# EVALUATION ONLY. Every cell reuses the SAME trained checkpoint (30k steps, seed 95)
# and only changes --num_input_views, so this costs no training and can co-reside
# with training on spare GPU memory.
#
# The point of the figure: LaCT's own Fig. (a) shows PSNR rising with input views and
# then flattening. If the rotary is doing what we claim (giving the fast-weight memory
# a usable address space), its advantage should GROW with the number of views, because
# more views means more content packed into the same fixed-size fast weights and
# therefore more pressure on addressing. A flat gap would mean the rotary buys a
# constant amount regardless of memory pressure, which would weaken the story.
#
# NOTE the models were TRAINED at 8 input views. Points far from 8 measure
# extrapolation, not in-distribution quality; say so in the caption.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
GPU=${1:-0}

# arm -> "run_dir config"
declare -A ARM_RUN=(
  [pra_hi]="pra_hi_s95 config/cam_pra_hi.yaml"
  [h_pra_hi]="h_pra_hi_s95 config/cam_h_pra_hi.yaml"
  [pra_h_hi]="pra_h_hi_s95 config/cam_pra_h_hi.yaml"
)
# NoPE has no surviving checkpoint on this node; it is queued as a training run.
ARMS=${ARMS:-"pra_hi h_pra_hi pra_h_hi"}
VIEWS=${VIEWS:-"2 4 8 16 24 32"}
SCENES=${SCENES:-256}
OUT=fig1_viewsweep
mkdir -p "$OUT"

echo "[fig1] gpu=$GPU arms='$ARMS' views='$VIEWS' scenes=$SCENES"
for arm in $ARMS; do
  read -r run cfg <<< "${ARM_RUN[$arm]}"
  ckpt="outputs/$run/model_0030000.pth"
  if [ ! -f "$ckpt" ]; then echo "[fig1] $arm: no $ckpt, skipping"; continue; fi
  for v in $VIEWS; do
    res="$OUT/${arm}_v${v}.json"
    if [ -f "$res" ]; then echo "[fig1] $arm v=$v already done"; continue; fi
    echo "[fig1] $arm v=$v ..."
    CUDA_VISIBLE_DEVICES=$GPU $PY eval.py \
        --load "$ckpt" --config "$cfg" \
        --num_scenes "$SCENES" --num_input_views "$v" --num_target_views 4 \
        --out "$res" > "$OUT/${arm}_v${v}.log" 2>&1 \
      || echo "[fig1] $arm v=$v FAILED (see $OUT/${arm}_v${v}.log)"
  done
done
echo "[fig1] done; results in $OUT/"
