#!/usr/bin/env bash
# DL3DV cell launcher (F50 protocol: 8-view 30k on /tmp/dl3dv, eval 140 scenes) with lock + eval, resumable.
#   ./run_dl3dv.sh <gpu> <exp> <config> [seed=95]
set -u
main() {
  GPU=$1; EXP=$2; CFG=$3; SEED=${4:-95}
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
  [ -f /tmp/dl3dv/train_index.json ] || { echo "FATAL: /tmp/dl3dv missing"; exit 1; }
  [ -f "outputs/$EXP/eval.json" ] && { echo "[$EXP] already evaluated"; exit 0; }
  mkdir -p outputs/.gpu_locks; LOCK="outputs/.gpu_locks/${NODE:-$(hostname -s)}_gpu$GPU"; echo "$EXP" > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
  CKPT="outputs/$EXP/model_0030000.pth"
  if [ ! -f "$CKPT" ]; then
    echo "[$EXP] train start $(date)  gpu=$GPU cfg=$CFG seed=$SEED (DL3DV 8-view 30k)"
    DATA_PATH=/tmp/dl3dv/train_index.json STEPS=30000 WARMUP=1500 bash launch_exp.sh "$GPU" "$EXP" "$CFG" "$SEED"
  fi
  [ -f "$CKPT" ] || { echo "[$EXP] FAILED: no checkpoint"; exit 1; }
  export TRITON_CACHE_DIR="$(cd .. && pwd)/.cache_triton_nvs" TORCHINDUCTOR_CACHE_DIR="$(cd .. && pwd)/.cache_inductor_nvs" TORCHINDUCTOR_COMPILE_THREADS=1
  CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "$CKPT" --config "$CFG" --data_path /tmp/dl3dv/test_index.json --num_scenes 140 > "outputs/$EXP/eval.log" 2>&1
  echo "[$EXP] eval exit=$? $(grep -h 'PSNR:' outputs/$EXP/eval.log | tail -1)"
}
main "$@"
