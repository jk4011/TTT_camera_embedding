#!/bin/bash
# When the Q30 d3 cells release GPUs 0-3, run the EVAL-ONLY view sweep of the F50
# 8-view-trained DL3DV checkpoints at v = {4, 16, 24, 32} (v=8 is F50 itself).
#
# Together with node2's Q34 (TRAINED at 32) this completes the 2x2 the user's
# hypothesis needs: if trained-at-32 composes but eval-only-at-32 does not, the
# recovery is phase adaptation during training, not evaluation-time geometry.
# F48 is the same sweep on tttLRM; this is the LVSM row.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1

# wait for the d3 workers to finish and for the test reshard
until [ "$(ls ../lact_llm/outputs/q30_base_d3/final.pt ../lact_llm/outputs/q30_in_d3_flat/final.pt ../lact_llm/outputs/q30_in_d3_nd/final.pt ../lact_llm/outputs/q30_h_d3_flat/final.pt ../lact_llm/outputs/q30_h_d3_nd/final.pt 2>/dev/null | wc -l)" -eq 5 ] \
      && [ -f /tmp/dl3dv/test_index.json ]; do sleep 300; done
echo "[$(date +%H:%M)] d3 done + test data ready -> eval sweep on GPUs 0-3"

declare -A CFG=( [base]=config/lact_l6_d256_p16.yaml [input]=config/cam_pra_hi.yaml
                 [hidden]=config/cam_h_pra_hi.yaml [both]=config/cam_pra_h_hi.yaml )
GPUS=(0 1 2 3); ARMS=(base input hidden both)
for i in 0 1 2 3; do
  (
    ARM=${ARMS[$i]}; GPU=${GPUS[$i]}
    for V in 4 16 24 32; do
      OUT="outputs/dl3dv_${ARM}_s95/eval_v${V}.json"
      [ -f "$OUT" ] && continue
      CUDA_VISIBLE_DEVICES=$GPU $PY eval.py \
        --load "outputs/dl3dv_${ARM}_s95/model_0030000.pth" --config "${CFG[$ARM]}" \
        --data_path /tmp/dl3dv/test_index.json --num_scenes 140 \
        --num_input_views $V --out "$OUT" \
        > "outputs/dl3dv_${ARM}_s95/eval_v${V}.log" 2>&1
      echo "[$(date +%H:%M)] $ARM v$V exit=$?"
    done
  ) &
done
wait
echo "[$(date +%H:%M)] eval sweep complete"
