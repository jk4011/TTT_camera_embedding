#!/bin/bash
# Table-1 SSIM sweep: re-evaluate every Table-1 cell with the SSIM-enabled eval.py,
# writing eval_ssim.json next to each run's original eval.json (originals untouched).
# Co-resides with f85 training; evals are small (<20 GB). Jobs striped over $GPUS.
set -u
GPUS=(${GPUS:-0 1})
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
cd "$(dirname "$0")"
# lustre compile caches + single-threaded inductor: the default /tmp cache is
# noexec and parallel compile subprocs crash (InductorError SubprocException)
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1

JOBS=(
  "base_s95|config/lact_l6_d256_p16.yaml|re10k"
  "q15_gta_in_s95|config/cam_gta_in.yaml|re10k"
  "q15_gta_in_s137|config/cam_gta_in.yaml|re10k"
  "q15_gta_in_s211|config/cam_gta_in.yaml|re10k"
  "q15_prope_orig_s95|config/cam_prope_orig.yaml|re10k"
  "q15_prope_orig_s137|config/cam_prope_orig.yaml|re10k"
  "q15_prope_orig_s211|config/cam_prope_orig.yaml|re10k"
  "pra_hi_s95|config/cam_pra_hi.yaml|re10k"
  "pra_hi_s137|config/cam_pra_hi.yaml|re10k"
  "pra_hi_s211|config/cam_pra_hi.yaml|re10k"
  "h_pra_hi_s95|config/cam_h_pra_hi.yaml|re10k"
  "h_pra_hi_s137|config/cam_h_pra_hi.yaml|re10k"
  "h_pra_hi_s211|config/cam_h_pra_hi.yaml|re10k"
  "pra_h_hi_s95|config/cam_pra_h_hi.yaml|re10k"
  "pra_h_hi_s137|config/cam_pra_h_hi.yaml|re10k"
  "pra_h_hi_s211|config/cam_pra_h_hi.yaml|re10k"
  "dl3dv32_base_s95|config/lact_l6_d256_p16.yaml|dl3dv32"
  "dl3dv32_input_s95|config/cam_pra_hi.yaml|dl3dv32"
  "dl3dv32_hidden_s95|config/cam_h_pra_hi.yaml|dl3dv32"
  "dl3dv32_both_s95|config/cam_pra_h_hi.yaml|dl3dv32"
)

run_lane() {
  local gpu=$1; shift
  for spec in "$@"; do
    IFS='|' read -r RUN CFG KIND <<< "$spec"
    [ -f "outputs/$RUN/eval_ssim.json" ] && { echo "[skip] $RUN"; continue; }
    EXTRA=""
    [ "$KIND" = "dl3dv32" ] && \
      EXTRA="--data_path /tmp/dl3dv/test_index.json --num_scenes 140 --num_input_views 32"
    echo "[gpu$gpu] $RUN"
    CUDA_VISIBLE_DEVICES=$gpu $PY eval.py \
      --load "outputs/$RUN/model_0030000.pth" --config "$CFG" $EXTRA \
      --out "outputs/$RUN/eval_ssim.json" > "outputs/$RUN/eval_ssim.log" 2>&1
    echo "[gpu$gpu] $RUN exit=$?"
  done
}

# stripe jobs across lanes
NL=${#GPUS[@]}
for i in "${!GPUS[@]}"; do
  LANE=()
  for j in "${!JOBS[@]}"; do [ $((j % NL)) -eq "$i" ] && LANE+=("${JOBS[$j]}"); done
  run_lane "${GPUS[$i]}" "${LANE[@]}" &
done
wait
echo "SSIM SWEEP DONE $(date)"
