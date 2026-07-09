#!/usr/bin/env bash
# Relaunch of the FULL ccv grid (2026-07-09): the pre-reset base/pra/both runs
# and their logs died with the T_B workspace, so all four variants restart
# together on the rebuilt data (lustre MultiCamVideo extraction, new pair index,
# same index_seed=42). Paired per-step comparison is valid WITHIN this grid
# (shared data order + deterministic per-step noise), which is all the analysis
# needs. ~46 h per run at 20k steps.
#
# Usage: bash run_ccv_grid.sh [sanity]
#   sanity — 60-step smoke runs (per IMPL_SPEC_CCV.md) instead of the 20k runs.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN_DIR="$REPO_ROOT/.venv_llm/bin"
NVS_OUT="$REPO_ROOT/lact_nvs/outputs"   # shared GPU-lock convention with queue_daemon
export HF_HOME=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/hf_cache
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
export C_INCLUDE_PATH=/usr/local/cuda/include
export PATH="/usr/local/cuda/bin:$PYTHON_BIN_DIR:$PATH"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor"
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$SCRIPT_DIR/outputs" "$NVS_OUT/.gpu_locks"

MODE=${1:-full}
declare -A GPUS=( [ccv_base]=2 [ccv_pra]=3 [ccv_both]=4 [ccv_pra_fixed]=5 )
declare -A CFGS=( [ccv_base]=abl_ccv_base [ccv_pra]=abl_ccv_pra [ccv_both]=abl_ccv_both [ccv_pra_fixed]=abl_ccv_pra_fixed )

cd "$SCRIPT_DIR/minVid"
for V in ccv_base ccv_pra ccv_both ccv_pra_fixed; do
  G=${GPUS[$V]}; CFG=${CFGS[$V]}
  if [ "$MODE" = "sanity" ]; then
    EXP="${V}_sanity"; EXTRA=(-s max_fwdbwd_passes 60 -s exp_name "$EXP")
  else
    EXP="$V"; EXTRA=(-s exp_name "$EXP")
  fi
  echo "ccv:$EXP" > "$NVS_OUT/.gpu_locks/gpu$G"   # claim GPU from the NVS daemon
  CUDA_VISIBLE_DEVICES=$G nohup "$PYTHON_BIN_DIR/python" -m torch.distributed.run \
    --standalone --nproc_per_node=1 --master_port $((29620 + G)) \
    train.py "configs/ar/${CFG}.yaml" "${EXTRA[@]}" \
    > "$SCRIPT_DIR/outputs/${EXP}.log" 2>&1 &
  echo "launched $EXP on gpu$G pid $! -> outputs/${EXP}.log"
done
