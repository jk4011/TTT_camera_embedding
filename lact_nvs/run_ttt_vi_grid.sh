#!/bin/bash
# Q46-TTT: the four-arm LaCT-LVSM grid on the NEW vary-intrinsics renders.
# Same recipe as run_gobj_grid.sh (F51) except data = /tmp/gobj_vi (24 views/obj,
# per-view random FOV/distance) and min_frames 24. If F51's "all rotary arms
# harmful at wide baseline" was the render distribution, this grid shows it in
# OUR stack. Usage: run_ttt_vi_grid.sh "<gpu0 gpu1 gpu2 gpu3>" [seed]
set -u
GPUS=${1:-"3 4 5 6"}
SEED=${2:-95}
cd "$(dirname "$0")"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1

declare -A CFG=(
  [base]=config/lact_l6_d256_p16.yaml
  [input]=config/cam_pra_hi.yaml
  [hidden]=config/cam_h_pra_hi.yaml
  [both]=config/cam_pra_h_hi.yaml
)
ARMS=(base input hidden both)
read -r -a G <<< "$GPUS"

run_one() {
  local GPU=$1 ARM=$2
  local EXP="gobjvi_${ARM}_s${SEED}"
  [ -f "outputs/$EXP/eval_ssim.json" ] && { echo "[$ARM] done"; return; }
  mkdir -p "outputs/$EXP"
  CUDA_VISIBLE_DEVICES=$GPU $PY -m torch.distributed.run \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
    train.py --config "${CFG[$ARM]}" \
    --data_path /tmp/gobj_vi/train_index.json --dataset re10k --scene_pose_normalize \
    --min_frames 24 \
    --expname "$EXP" \
    --steps 30000 --warmup 1500 --lr 1e-4 --lpips_start 5000 --seed "$SEED" \
    --bs_per_gpu 16 --num_all_views 15 --num_input_views 8 --num_target_views 8 \
    --image_size 256 256 --num_workers 7 \
    --save_every 10000 --log_every 200 \
    > "outputs/$EXP/train.log" 2>&1
  echo "EXIT $? $EXP" >> outputs/exp_status.log
  CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$EXP/model_0030000.pth" \
    --config "${CFG[$ARM]}" --data_path /tmp/gobj_vi/test_index.json \
    --num_scenes 500 --min_frames 24 \
    --out "outputs/$EXP/eval_ssim.json" > "outputs/$EXP/eval.log" 2>&1
  echo "[$ARM] eval exit=$?"
}
for i in "${!ARMS[@]}"; do run_one "${G[$i]}" "${ARMS[$i]}" & sleep 5; done
wait
echo "TTT-VI GRID DONE $(date)"
