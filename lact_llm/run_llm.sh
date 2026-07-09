#!/usr/bin/env bash
# Usage: ./run_llm.sh <gpu> <out_name> [extra train_small.py args...]
# Example: ./run_llm.sh 0 base_small --token_budget 500000000 --bs 24
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <gpu> <out_name> [extra args...]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_TOP="$(dirname "$SCRIPT_DIR")"
PYTHON="$REPO_TOP/.venv_llm/bin/python"

export HF_HOME=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/hf_cache
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
export C_INCLUDE_PATH=/usr/local/cuda/include
export PATH=/usr/local/cuda/bin:$PATH
# /tmp and /dev/shm are noexec here; triton needs an exec-allowed cache dir
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor"

mkdir -p "outputs/$2"

CUDA_VISIBLE_DEVICES=$1 "$PYTHON" train_small.py \
    --out_dir "outputs/$2" \
    "${@:3}" > "outputs/$2/train.log" 2>&1
