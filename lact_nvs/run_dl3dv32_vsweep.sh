#!/bin/bash
# Fig.2 middle panel (user 2026-08-12): the 32-view-TRAINED dl3dv32 arms evaluated
# at {8,16,24,64} input views (32 = the existing eval.json). Single lane on $GPU.
set -u
GPU=${GPU:-3}
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
declare -A CFG=( [base]=config/lact_l6_d256_p16.yaml [input]=config/cam_pra_hi.yaml
                 [hidden]=config/cam_h_pra_hi.yaml [both]=config/cam_pra_h_hi.yaml )
for V in 8 16 24 64; do
  for ARM in base input hidden both; do
    OUT="outputs/dl3dv32_${ARM}_s95/eval_v${V}.json"
    [ -f "$OUT" ] && continue
    BS=8; [ "$V" -ge 64 ] && BS=4
    CUDA_VISIBLE_DEVICES=$GPU $PY eval.py \
      --load "outputs/dl3dv32_${ARM}_s95/model_0030000.pth" --config "${CFG[$ARM]}" \
      --data_path /tmp/dl3dv/test_index.json --num_scenes 140 \
      --num_input_views $V --bs $BS --out "$OUT" \
      > "outputs/dl3dv32_${ARM}_s95/eval_v${V}.log" 2>&1
    echo "[v$V $ARM] exit=$? scenes=$($PY -c "import json;print(json.load(open('$OUT'))['num_scenes'])" 2>/dev/null)"
  done
done
echo "DL3DV32 VSWEEP DONE $(date)"
