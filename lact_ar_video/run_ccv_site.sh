#!/usr/bin/env bash
# P5 CCV site ablation (NODE2_PROMPT.md 2026-08-05): one cell per GPU, run in parallel.
# Usage: ./run_ccv_site.sh <cell: in|h|both> <gpu>
#
# All three cells are cam_encoder ON + FIXED ladder (ttt_learnable_freqs: false),
# differing only in the two site flags, so the ablation lives in the same family as
# the headline ccv_base vs ccv_both comparison. Configs: configs/ar/abl_ccv_site_*.yaml
#
# master_port is derived from the GPU id (NOT a loop index, which is always 0 and
# gives every cell the same port -> EADDRINUSE; node1 hit this today).
set -uo pipefail

CELL="$1"; GPU="$2"
EXP="ccv_site_${CELL}"
CFG="configs/ar/abl_ccv_site_${CELL}.yaml"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN_DIR="$REPO_ROOT/.venv_llm/bin"

export HF_HOME=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/hf_cache
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
export C_INCLUDE_PATH=/usr/local/cuda/include
export PATH="/usr/local/cuda/bin:$PYTHON_BIN_DIR:$PATH"
# /tmp is noexec here; triton must compile into the repo.
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor"
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$SCRIPT_DIR/outputs"

cd "$SCRIPT_DIR/minVid"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN_DIR/python" -m torch.distributed.run \
    --standalone --nproc_per_node=1 --master_port $((29700 + GPU)) \
    train.py "$CFG" -s exp_name "$EXP" \
    >> "$SCRIPT_DIR/outputs/${EXP}.log" 2>&1
echo "EXIT $? $EXP"
