#!/usr/bin/env bash
# ccv sanity: 60 training steps for each of the three camera-controlled
# video variants (base / pra / both) on GPU 3 ONLY, batch 1.
# Logs to /tmp/ccv_sanity_<variant>.log.
# Usage: bash run_ccv_sanity.sh [variant ...]   (default: base pra both)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN_DIR="$REPO_ROOT/.venv_llm/bin"

# /tmp is noexec: compiled-extension caches must live on the repo filesystem.
export HF_HOME=/tmp/hf_cache
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
export C_INCLUDE_PATH=/usr/local/cuda/include
export PATH="/usr/local/cuda/bin:$PYTHON_BIN_DIR:$PATH"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$SCRIPT_DIR/minVid"

# HARD CONSTRAINT: GPUs 0,1,2 run live training. GPU 3 only.
export CUDA_VISIBLE_DEVICES=3
export NCCL_DEBUG=WARN

VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
    VARIANTS=(base pra both)
fi

for v in "${VARIANTS[@]}"; do
    LOG="/tmp/ccv_sanity_${v}.log"
    echo "=== ccv sanity: variant $v -> $LOG ==="
    "$PYTHON_BIN_DIR/python" -m torch.distributed.run \
        --standalone --nproc_per_node=1 \
        train.py "configs/ar/abl_ccv_${v}.yaml" \
        -s exp_name "abl_ccv_${v}_sanity" \
        -s max_fwdbwd_passes 60 \
        > "$LOG" 2>&1
    status=$?
    echo "variant $v exited with status $status"
    grep -E "^step |loss:" "$LOG" | tail -3
done
