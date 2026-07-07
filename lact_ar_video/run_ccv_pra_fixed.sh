#!/usr/bin/env bash
# Q5: clean ccv run (TTT-RoPE, fixed ladders, no learnable freqs) on GPU 3.
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
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$SCRIPT_DIR/minVid"
CUDA_VISIBLE_DEVICES=3 nohup "$PYTHON_BIN_DIR/python" -m torch.distributed.run \
  --standalone --nproc_per_node=1 --master_port 29620 \
  train.py "configs/ar/abl_ccv_pra_fixed.yaml" \
  -s exp_name "ccv_pra_fixed" \
  > /tmp/ccv_pra_fixed.log 2>&1 &
echo "launched ccv_pra_fixed pid $!"
