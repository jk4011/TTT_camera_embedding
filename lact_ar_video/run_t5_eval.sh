#!/bin/bash
# Held-out paired val loss for the 4 t5 arms (one arm per GPU, 64 clips each).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/minVid"
PY=../../.venv_llm/bin/python
export PYTHONPATH="$(cd .. && pwd)"
export TRITON_CACHE_DIR=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/.cache_triton_tttlrm
export TORCHINDUCTOR_CACHE_DIR=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/.cache_inductor_tttlrm
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p ../outputs/eval_dev
i=0
for arm in base in h both; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup $PY eval_video2_valloss.py \
    --config configs/ar/video2_t5_${arm}.yaml \
    --ckpt ../outputs/video2_t5_${arm}/seed_1/checkpoint_model_001499 \
    --out ../outputs/eval_dev/valloss_video2_t5_${arm}_1499.json \
    > ../outputs/eval_dev/valloss_video2_t5_${arm}.log 2>&1 < /dev/null &
  i=$((i+1))
done
echo "4 eval jobs launched"
