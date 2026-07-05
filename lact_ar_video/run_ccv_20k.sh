#!/usr/bin/env bash
# Launch the three 20k-step camera-controlled video runs, one per GPU (0,1,2).
# Run this ONLY after the v20k runs have finished and GPUs 0-2 are free.
# Usage: bash run_ccv_20k.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN_DIR="$REPO_ROOT/.venv_llm/bin"

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

# refuse to launch if GPUs 0-2 are still busy
BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1<3 && $2>2000 {print $1}')
if [ -n "$BUSY" ]; then
    echo "GPUs still busy: $BUSY -- aborting launch." >&2
    exit 1
fi

GPUS=(0 1 2)
VARIANTS=(base pra both)
for i in 0 1 2; do
    v="${VARIANTS[$i]}"
    LOG="/tmp/ccv_${v}.log"
    echo "=== launching ccv_${v} on GPU ${GPUS[$i]} -> $LOG ==="
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" nohup \
        "$PYTHON_BIN_DIR/python" -m torch.distributed.run \
        --standalone --nproc_per_node=1 --master_port $((29610 + i)) \
        train.py "configs/ar/abl_ccv_${v}.yaml" \
        -s exp_name "ccv_${v}" \
        > "$LOG" 2>&1 &
    echo "pid $!"
    sleep 5
done
echo "all launched; logs at /tmp/ccv_{base,pra,both}.log"
