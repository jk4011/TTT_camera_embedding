#!/usr/bin/env bash
# Q33: LaCT-LVSM on gobjaverse (object orbits) -- the third point on the geometry axis.
#
#   ./run_gobj_grid.sh "0 1 2 3" [seed]
#
# RE10K 7.2 deg -> DL3DV 61.1 deg -> gobjaverse 93.2 deg median between-view angle
# (measured through the actual loader). If `both`'s failure to compose tracks this
# axis monotonically, "the two sites compose at narrow baselines and interfere at wide
# ones" becomes a dose-response claim instead of a two-point contrast.
#
# Same protocol as run_dl3dv_grid.sh with two forced deviations, shared by all arms:
#   --min_frames 40   gobjaverse scenes have exactly 40 frames, below the default
#                     num_views*3 = 45 filter, which would silently drop EVERY scene
#                     (the dataset would be empty and the run would crash, which is
#                     at least loud; do not remove this flag).
#   held-out split    gobjaverse has no official test split here; the last 500 keys
#                     (sorted) are held out at reshard time by run instructions.
set -u
GPUS=${1:-"0 1 2 3"}; SEED=${2:-95}
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1

[ -f /tmp/gobj/train_index.json ] || { echo "FATAL: /tmp/gobj/train_index.json missing"; exit 1; }
[ -f /tmp/gobj/test_index.json ]  || { echo "FATAL: test index missing"; exit 1; }

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
  local EXP="gobj_${ARM}_s${SEED}"
  [ -f "outputs/$EXP/eval.json" ] && { echo "[$ARM] done"; return; }
  # launch_exp.sh does not know --min_frames, so train directly with its recipe
  mkdir -p "outputs/$EXP"
  CUDA_VISIBLE_DEVICES=$GPU $PY -m torch.distributed.run \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
    train.py --config "${CFG[$ARM]}" \
    --data_path /tmp/gobj/train_index.json --dataset re10k --scene_pose_normalize \
    --min_frames 40 \
    --expname "$EXP" \
    --steps 30000 --warmup 1500 --lr 1e-4 --lpips_start 5000 --seed "$SEED" \
    --bs_per_gpu 16 --num_all_views 15 --num_input_views 8 --num_target_views 8 \
    --image_size 256 256 --num_workers 7 \
    --save_every 10000 --log_every 200 \
    > "outputs/$EXP/train.log" 2>&1
  echo "EXIT $? $EXP" >> outputs/exp_status.log
  CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$EXP/model_0030000.pth" \
    --config "${CFG[$ARM]}" --data_path /tmp/gobj/test_index.json --num_scenes 500 \
    > "outputs/$EXP/eval.log" 2>&1
  echo "[$ARM] eval exit=$?"
}
for i in "${!ARMS[@]}"; do run_one "${G[$i]}" "${ARMS[$i]}" & sleep 5; done
wait
echo "[gobj grid] all four arms done"
