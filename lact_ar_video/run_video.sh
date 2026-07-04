#!/usr/bin/env bash
# Usage: ./run_video.sh <gpu_ids> <config_path_relative_to_minVid> [extra train.py args...]
# Example: ./run_video.sh 0 configs/ar/ablation_small.yaml
#          ./run_video.sh 0 configs/ar/ablation_small.yaml -s max_fwdbwd_passes 10
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <gpu_ids (e.g. 0 or 0,1)> <config.yaml> [extra args...]" >&2
    exit 1
fi

GPU_IDS="$1"
CONFIG="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON_BIN_DIR="$REPO_ROOT/.venv_llm/bin"

# Weights / data caches on tmpfs are fine; compiled-extension caches are NOT
# (/tmp is noexec here) -> keep triton/inductor caches inside the repo.
export HF_HOME=/tmp/hf_cache
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
export C_INCLUDE_PATH=/usr/local/cuda/include
export PATH="/usr/local/cuda/bin:$PYTHON_BIN_DIR:$PATH"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# minVid imports use both `minVid.*` (needs lact_ar_video on path) and
# top-level `models.*` in configs (needs cwd == minVid dir).
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$SCRIPT_DIR/minVid"

NPROC=$(awk -F',' '{print NF}' <<< "$GPU_IDS")

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NCCL_DEBUG=WARN

exec "$PYTHON_BIN_DIR/python" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC" \
    train.py "$CONFIG" "$@"
