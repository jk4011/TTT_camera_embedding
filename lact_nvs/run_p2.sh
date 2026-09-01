#!/usr/bin/env bash
# Protocol-2 launcher (2026-09-01, user directive): **2 input views, 80k steps** -- the setting of the prior
# camera-PE papers (pixelSplat/MVSplat/PRoPE-style 2-view RE10K).
#
#   ./run_p2.sh <gpu> <exp> <config> [seed=95]
#   env: STEPS (80000), WARMUP (4000), NODE (hostname -s), SKIP_EVAL=1,
#        DATA=re10k (default, /tmp/re10k) | gobj_vi (/tmp/gobj_vi, min_frames 24) | dl3dv (/tmp/dl3dv)
#
# Train: num_all_views 6 = 2 inputs + 4 targets sampled from a random window (loader default: w in
#        [18,192] frames), bs16, lr 1e-4, LPIPS from 5k, 256x256, seed-pinned.
# Eval:  deterministic 2 inputs at the ends of a centred 90-frame window (approx. the pixelSplat test
#        context gap), 4 midpoint targets; RE10K 256 held-out scenes / gobj_vi 500 / dl3dv 140.
# Resumable: skips a finished cell (eval.json); train.py resumes from its outputs dir.
set -u
main() {
  GPU=$1; EXP=$2; CFG=$3; SEED=${4:-95}
  STEPS=${STEPS:-80000}; WARMUP=${WARMUP:-4000}
  DATA=${DATA:-re10k}
  case "$DATA" in
    re10k)   TRAIN=/tmp/re10k/train_index.json;   TEST=/tmp/re10k/test_index.json;   NSC=256; MINF=() ;;
    gobj_vi) TRAIN=/tmp/gobj_vi/train_index.json; TEST=/tmp/gobj_vi/test_index.json; NSC=500; MINF=(--min_frames 24) ;;
    dl3dv)   TRAIN=/tmp/dl3dv/train_index.json;   TEST=/tmp/dl3dv/test_index.json;   NSC=140; MINF=() ;;
    *) echo "FATAL: DATA must be re10k|gobj_vi|dl3dv"; exit 1 ;;
  esac
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
  REPO_ROOT="$(cd .. && pwd)"
  export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs" TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs" TORCHINDUCTOR_COMPILE_THREADS=1
  mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" outputs/.gpu_locks
  [ -f "$TRAIN" ] || { echo "FATAL: $TRAIN missing (reshard first)"; exit 1; }
  [ -f "outputs/$EXP/eval.json" ] && { echo "[$EXP] already evaluated"; exit 0; }
  LOCK="outputs/.gpu_locks/${NODE:-$(hostname -s)}_gpu$GPU"; echo "$EXP" > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
  mkdir -p "outputs/$EXP"
  CKPT="outputs/$EXP/model_$(printf %07d "$STEPS").pth"
  if [ ! -f "$CKPT" ]; then
    echo "[$EXP] train start $(date)  gpu=$GPU cfg=$CFG seed=$SEED steps=$STEPS data=$DATA views=2+4"
    CUDA_VISIBLE_DEVICES=$GPU $PY -m torch.distributed.run \
      --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
      train.py --config "$CFG" \
      --data_path "$TRAIN" --dataset re10k --scene_pose_normalize "${MINF[@]}" \
      --expname "$EXP" \
      --steps "$STEPS" --warmup "$WARMUP" --lr 1e-4 --lpips_start 5000 --seed "$SEED" \
      --bs_per_gpu 16 --num_all_views 6 --num_input_views 2 --num_target_views 4 \
      --image_size 256 256 --num_workers 7 \
      --save_every 10000 --log_every 200 \
      >> "outputs/$EXP/train.log" 2>&1
    echo "EXIT $? $EXP" >> outputs/exp_status.log
  fi
  [ -f "$CKPT" ] || { echo "[$EXP] FAILED: no checkpoint $CKPT"; exit 1; }
  [ "${SKIP_EVAL:-0}" = "1" ] && exit 0
  echo "[$EXP] eval start $(date)"
  CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "$CKPT" --config "$CFG" \
    --data_path "$TEST" --num_scenes $NSC "${MINF[@]}" \
    --num_input_views 2 --num_target_views 4 --window 90 \
    > "outputs/$EXP/eval.log" 2>&1
  echo "[$EXP] eval exit=$? $(grep -h 'PSNR:' outputs/$EXP/eval.log | tail -1)"
}
main "$@"
