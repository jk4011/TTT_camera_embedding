#!/bin/bash
# Resume-train one NoPE baseline seed (Table-1 NoPE row SSIM) + SSIM eval.
# Usage: run_base_seed_chain.sh <gpu> <seed>
set -u
GPU=$1; SEED=$2
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
cd "$(dirname "$0")"
until [ -f /tmp/re10k/train_index.json ]; do sleep 120; done
until [ -f /tmp/re10k/test_index.json ]; do sleep 120; done
echo "base_s${SEED}" > outputs/.gpu_locks/$(hostname -s)_gpu${GPU}
bash launch_exp.sh "$GPU" "base_s${SEED}" config/lact_l6_d256_p16.yaml "$SEED"
CUDA_VISIBLE_DEVICES=$GPU $PY eval.py \
  --load "outputs/base_s${SEED}/model_0030000.pth" \
  --config config/lact_l6_d256_p16.yaml \
  --out "outputs/base_s${SEED}/eval_ssim.json" \
  > "outputs/base_s${SEED}/eval_ssim.log" 2>&1
echo "[base_s${SEED}] eval exit=$?"
rm -f outputs/.gpu_locks/$(hostname -s)_gpu${GPU}
