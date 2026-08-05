#!/bin/bash
# Does TTT-RoPE survive an n-step update? EVALUATION ONLY.
#
#   bash run_chunk_sweep.sh 0            # all arms, n in {1,2,4,8,16}, 32 input views
#   CHUNKS="1 4" ARMS="pra_h_hi" bash run_chunk_sweep.sh 3
#
# The method is derived for a SINGLE update step: the phases cancel inside one inner
# product, one gradient step. At scale the update is split into n sequential chunks,
# each updating the fast weights on top of the previous chunk's result, and the
# derivation does not cover that. tttLRM already runs this way (full_ttt_op steps
# through update_minibatch = 1024) while NVS does one update over all input tokens.
#
# These checkpoints were TRAINED with a single update. Evaluating them at n > 1 asks a
# narrower question than retraining would: do the learned phases still address correctly
# when the update is chunked? A positive answer supports the extension; it is not the
# same as showing the method trains well in the n-step regime, and the writeup must say
# which of the two it is.
#
# The quantity of interest is Delta(rotary - NoPE) AT EACH n, not absolute PSNR across
# n. Absolute quality is expected to fall as chunks shrink, for reasons that have
# nothing to do with the rotary (per-chunk weight-norm decays earlier chunks).
#
# 32 input views = 8192 update tokens, so chunk size is 8192/n:
#   n=1 8192 | n=2 4096 | n=4 2048 | n=8 1024 | n=16 512 | n=32 256
# n=32 falls below Muon's ~427-token amortisation point, which is the confound F8 hit
# (-0.23 dB from chunk size alone), so it is excluded by default.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
GPU=${1:-0}

REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

declare -A ARM_RUN=(
  [base]="base_s95 config/lact_l6_d256_p16.yaml"
  [pra_hi]="pra_hi_s95 config/cam_pra_hi.yaml"
  [h_pra_hi]="h_pra_hi_s95 config/cam_h_pra_hi.yaml"
  [pra_h_hi]="pra_h_hi_s95 config/cam_pra_h_hi.yaml"
)
ARMS=${ARMS:-"base pra_hi h_pra_hi pra_h_hi"}
CHUNKS=${CHUNKS:-"1 2 4 8 16"}
VIEWS=${VIEWS:-32}
SCENES=${SCENES:-256}
OUT=chunk_sweep
mkdir -p "$OUT"

echo "[chunk] gpu=$GPU arms='$ARMS' chunks='$CHUNKS' views=$VIEWS scenes=$SCENES"
for arm in $ARMS; do
  read -r run cfg <<< "${ARM_RUN[$arm]}"
  ckpt="outputs/$run/model_0030000.pth"
  [ -f "$ckpt" ] || { echo "[chunk] $arm: no $ckpt, skipping"; continue; }
  for n in $CHUNKS; do
    res="$OUT/${arm}_n${n}.json"
    [ -f "$res" ] && { echo "[chunk] $arm n=$n already done"; continue; }
    echo "[chunk] $arm n=$n (chunk size $((VIEWS*256/n)) tokens) ..."
    CUDA_VISIBLE_DEVICES=$GPU $PY eval.py \
        --load "$ckpt" --config "$cfg" \
        --num_scenes "$SCENES" --num_input_views "$VIEWS" --num_target_views 4 \
        --ttt_num_chunks "$n" --out "$res" \
      > "$OUT/${arm}_n${n}.log" 2>&1 \
      || echo "[chunk] $arm n=$n FAILED (see $OUT/${arm}_n${n}.log)"
  done
done
echo "[chunk] done; results in $OUT/"
