#!/bin/bash
# Fig.2 DL3DV panel: add the 64-view evaluation point (8-view-trained F50 arms).
set -u
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
declare -A CFG=( [base]=config/lact_l6_d256_p16.yaml [input]=config/cam_pra_hi.yaml
                 [hidden]=config/cam_h_pra_hi.yaml [both]=config/cam_pra_h_hi.yaml )
lane() {
  local GPU=$1; shift
  for ARM in "$@"; do
    OUT="outputs/dl3dv_${ARM}_s95/eval_v64.json"
    [ -f "$OUT" ] && continue
    CUDA_VISIBLE_DEVICES=$GPU $PY eval.py \
      --load "outputs/dl3dv_${ARM}_s95/model_0030000.pth" --config "${CFG[$ARM]}" \
      --data_path /tmp/dl3dv/test_index.json --num_scenes 140 \
      --num_input_views 64 --bs 4 --out "$OUT" \
      > "outputs/dl3dv_${ARM}_s95/eval_v64.log" 2>&1
    echo "[$ARM v64] exit=$? scenes=$($PY -c "import json;print(json.load(open('$OUT'))['num_scenes'])" 2>/dev/null)"
  done
}
lane 0 base input &
lane 1 hidden both &
wait
echo "V64 SWEEP DONE $(date)"
