#!/usr/bin/env bash
# Input-view sweep evaluation (2026-09-01, user request): 4 arms x views {4,8,12,20,32,48} on one dataset.
#   ./run_vsweep.sh <dataset: re10k|dl3dvw48|gobjv60> <gpu> [views="4 8 12 20 32 48"]
# Arms: base / input (qk_rope_cam) / hidden (h_pra) / both (TTT-RoPE) -- 8-view-trained checkpoints:
#   re10k    -> {base,pra_hi,h_pra_hi,pra_h_hi}_s137            (256 scenes with >=52 frames, window 128)
#   dl3dvw48 -> dl3dvw48_{base,input,hidden,both}_s137, 256x448  (140 scenes)
#   gobjv60  -> gobjvi_{base,input,hidden,both}_s95 evaluated on the NEW 60-view renders (/tmp/gobj_v60, 501 objects)
# Writes outputs/<exp>/eval_nv<V>.json (+ .log); skips finished ones. A fixed --min_frames keeps the scene set
# identical across V so per-scene paired deltas are valid.
set -u
DS=$1; GPU=$2; VIEWS=${3:-"4 8 12 20 32 48"}
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"; export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs" TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs" TORCHINDUCTOR_COMPILE_THREADS=1
declare -A CFG=( [base]=config/lact_l6_d256_p16.yaml [input]=config/cam_pra_hi.yaml [hidden]=config/cam_h_pra_hi.yaml [both]=config/cam_pra_h_hi.yaml )
case "$DS" in
  re10k)    declare -A EXP=( [base]=base_s137 [input]=pra_hi_s137 [hidden]=h_pra_hi_s137 [both]=pra_h_hi_s137 ); DP=/tmp/re10k/test_index.json; NSC=256; EXTRA=(--min_frames 52) ;;
  dl3dvw48) declare -A EXP=( [base]=dl3dvw48_base_s137 [input]=dl3dvw48_input_s137 [hidden]=dl3dvw48_hidden_s137 [both]=dl3dvw48_both_s137 ); DP=/tmp/dl3dv/test_index.json; NSC=140; EXTRA=(--min_frames 52 --image_size 256 448) ;;
  gobjv60)  declare -A EXP=( [base]=gobjvi_base_s95 [input]=gobjvi_input_s95 [hidden]=gobjvi_hidden_s95 [both]=gobjvi_both_s95 ); DP=/tmp/gobj_v60/test_index.json; NSC=500; EXTRA=(--min_frames 60) ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
[ -f "$DP" ] || { echo "FATAL: $DP missing"; exit 1; }
for V in $VIEWS; do
  if [ "$V" -ge 32 ]; then BS=2; elif [ "$V" -ge 20 ]; then BS=4; else BS=8; fi
  for ARM in base input hidden both; do
    E=${EXP[$ARM]}; OUT="outputs/$E/eval_${DS}_nv${V}.json"
    [ -f "$OUT" ] && { echo "[$DS gpu$GPU] $ARM v$V done"; continue; }
    echo "[$DS gpu$GPU] $ARM v$V start $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$E/model_0030000.pth" --config "${CFG[$ARM]}" \
      --data_path "$DP" --num_scenes $NSC "${EXTRA[@]}" --num_input_views "$V" --num_target_views 4 --bs $BS \
      --out "$OUT" > "outputs/$E/eval_${DS}_nv${V}.log" 2>&1
    echo "[$DS gpu$GPU] $ARM v$V exit=$? $(grep -h 'PSNR:' outputs/$E/eval_${DS}_nv${V}.log | tail -1)"
  done
done
echo "[$DS] SWEEP DONE"
