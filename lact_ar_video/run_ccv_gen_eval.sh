#!/usr/bin/env bash
# CCV sampled-metrics eval (queued 2026-08-12): generate all 64 held-out pairs per
# F54 cell with fixed seeds and score PSNR/SSIM/LPIPS vs GT. Fills the evaluation-
# metric columns of the paper's CCV table (currently val loss only).
#
# 8 jobs = 4 cells x 2 shards (pairs 0-31 / 32-63), ~9.5 h per job at 40 Euler
# steps (~18 min/pair). Stripe over the GPUs given on the command line:
#   ./run_ccv_gen_eval.sh 0 1 2 3 4 5 6 7    # 8 GPUs -> everything parallel, ~10 h
#   ./run_ccv_gen_eval.sh 0 1 2 3            # 4 GPUs -> two rounds, ~19 h
# Per-pair flush + crash resume are built into eval_ccv_generate.py; rerunning
# this script after an interruption continues where it stopped.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ $# -ge 1 ] || { echo "usage: $0 <gpu> [gpu ...]"; exit 1; }
GPUS=("$@")

REPO_ROOT="$(cd .. && pwd)"
PY="$REPO_ROOT/.venv_llm/bin/python"
export HF_HOME=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/hf_cache
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
export C_INCLUDE_PATH=/usr/local/cuda/include
export PATH="/usr/local/cuda/bin:$(dirname "$PY"):$PATH"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor"
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH="$PWD"

AV_ROOT="$PWD"  # lact_ar_video; captured before any cd (lane cd's into minVid)

ckpt_for() {
  local EXP=$1
  local A="$AV_ROOT/outputs/_keep_step13999/${EXP}_checkpoint_model_013999"
  local L="$AV_ROOT/outputs/${EXP}/seed_1/checkpoint_model_013999"
  if [ -d "$A" ]; then echo "$A"; else echo "$L"; fi
}

# cell|config|exp_name
CELLS=(
  "base|configs/ar/abl_ccv_base.yaml|ccv_base"
  "in|configs/ar/abl_ccv_site_in.yaml|ccv_site_in"
  "h|configs/ar/abl_ccv_site_h.yaml|ccv_site_h"
  "both|configs/ar/abl_ccv_site_both.yaml|ccv_site_both"
)
JOBS=()
for spec in "${CELLS[@]}"; do
  JOBS+=("$spec|0" "$spec|32")
done

lane() {
  # eval_ccv_* resolve the pairs json and configs relative to minVid, matching
  # run_ccv_site_eval.sh: run from inside minVid, outputs one level up.
  local gpu=$1; shift
  for job in "$@"; do
    IFS='|' read -r CELL CFG EXP START <<< "$job"
    OUT="$PWD/outputs/eval_site/gen_${EXP}_013999"
    mkdir -p "$OUT"
    echo "[gpu$gpu] $EXP shard start=$START"
    ( cd minVid && CUDA_VISIBLE_DEVICES=$gpu $PY eval_ccv_generate.py \
        --config "$CFG" --ckpt "$(ckpt_for "$EXP")" \
        --out "$OUT" --start "$START" --n_pairs 32 --steps 40 \
        > "$OUT/gen_start${START}.log" 2>&1 )
    echo "[gpu$gpu] $EXP shard start=$START exit=$?"
  done
}

NL=${#GPUS[@]}
for i in "${!GPUS[@]}"; do
  LANE=()
  for j in "${!JOBS[@]}"; do [ $((j % NL)) -eq "$i" ] && LANE+=("${JOBS[$j]}"); done
  lane "${GPUS[$i]}" "${LANE[@]}" &
done
wait
echo "CCV GEN EVAL DONE $(date)"
