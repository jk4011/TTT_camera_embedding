#!/usr/bin/env bash
# Generic single-cell launcher for the gObjaverse camera-embedding program (2026-08-31).
#
#   ./run_gobj.sh <gpu> <exp> <config> [seed=95]
#   env: STEPS (30000), DEPTH_DIR (unset; set for oracle cells -> --depth_dir on train+eval),
#        NODE (hostname -s) for the shared lock dir, SKIP_EVAL=1,
#        DATA=gobj (default; orbit renders, /tmp/gobj, min_frames 40) | gobj_vi (RayRoPE
#        vary-intrinsics re-renders, /tmp/gobj_vi, 24 views/object, min_frames 24 -- the
#        F69 / run_ttt_vi_grid.sh protocol; exp names conventionally start with gobjvi_)
#
# Protocol = run_gobj_grid.sh exactly (F51 recipe): RE10K-format gObjaverse in /tmp/gobj,
# --scene_pose_normalize, --min_frames 40, 30k iters, bs16, lr 1e-4, LPIPS from 5k, 8 input
# + 8 target of 15... (num_all_views 15 is the stock launcher value; the loader samples
# num_all_views frames and the trainer slices 8 inputs / last 8 targets), 256x256.
# Eval: /tmp/gobj/test_index.json, first 500 scenes, 8 uniform inputs / 4 midpoint targets.
# Resumable: skips a finished cell (eval.json), train.py resumes from its outputs dir.
set -u
GPU=$1; EXP=$2; CFG=$3; SEED=${4:-95}
STEPS=${STEPS:-30000}
DATA=${DATA:-gobj}
case "$DATA" in
  gobj)    DROOT=/tmp/gobj;    MINF=40 ;;
  gobj_vi) DROOT=/tmp/gobj_vi; MINF=24 ;;
  *) echo "FATAL: DATA must be gobj or gobj_vi"; exit 1 ;;
esac
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" outputs/.gpu_locks
[ -f $DROOT/train_index.json ] || { echo "FATAL: $DROOT/train_index.json missing (reshard_gobjaverse.py / reshard_rayrope_renders.py)"; exit 1; }
[ -f $DROOT/test_index.json ]  || { echo "FATAL: $DROOT test index missing"; exit 1; }
[ -f "outputs/$EXP/eval.json" ] && { echo "[$EXP] already evaluated"; exit 0; }

DEPTH_ARGS=()
if [ -n "${DEPTH_DIR:-}" ]; then
  DEPTH_TRAIN=(--depth_dir "$DEPTH_DIR/train"); DEPTH_EVAL=(--depth_dir "$DEPTH_DIR/test")
else
  DEPTH_TRAIN=(); DEPTH_EVAL=()
fi
LOCK="outputs/.gpu_locks/${NODE:-$(hostname -s)}_gpu$GPU"
echo "$EXP" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

mkdir -p "outputs/$EXP"
CKPT="outputs/$EXP/model_$(printf %07d "$STEPS").pth"
if [ ! -f "$CKPT" ]; then
  echo "[$EXP] train start $(date)  gpu=$GPU cfg=$CFG seed=$SEED steps=$STEPS data=$DATA depth=${DEPTH_DIR:-none}"
  CUDA_VISIBLE_DEVICES=$GPU $PY -m torch.distributed.run \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
    train.py --config "$CFG" \
    --data_path $DROOT/train_index.json --dataset re10k --scene_pose_normalize \
    --min_frames $MINF "${DEPTH_TRAIN[@]}" \
    --expname "$EXP" \
    --steps "$STEPS" --warmup 1500 --lr 1e-4 --lpips_start 5000 --seed "$SEED" \
    --bs_per_gpu 16 --num_all_views 15 --num_input_views 8 --num_target_views 8 \
    --image_size 256 256 --num_workers 7 \
    --save_every 10000 --log_every 200 \
    >> "outputs/$EXP/train.log" 2>&1
  echo "EXIT $? $EXP" >> outputs/exp_status.log
fi
[ -f "$CKPT" ] || { echo "[$EXP] FAILED: no checkpoint $CKPT"; exit 1; }
[ "${SKIP_EVAL:-0}" = "1" ] && exit 0
echo "[$EXP] eval start $(date)"
CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "$CKPT" --config "$CFG" \
  --data_path $DROOT/test_index.json --num_scenes 500 --min_frames $MINF "${DEPTH_EVAL[@]}" \
  > "outputs/$EXP/eval.log" 2>&1
echo "[$EXP] eval exit=$? $(grep -h 'PSNR:' outputs/$EXP/eval.log | tail -1)"
