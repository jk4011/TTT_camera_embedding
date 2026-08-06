#!/usr/bin/env bash
# Q34b: TRAINING-time camera baseline on DL3DV. Four arms at a narrow training window.
#
#   ./run_dl3dv_window.sh "0 1 2 3" [window] [seed]
#
# F52 established the dose-response at EVAL time: on one dataset, one model, widening
# the view window from 5.7 to 47.8 deg drives both - max(single) from ~0 to -0.16,
# monotone and replicated across two seeds. That says the effect is a property of the
# camera geometry the model is TESTED on. This grid asks the other half: does training
# at a narrow baseline change what the model LEARNS about composing the two sites?
#
# The existing dl3dv_*_s95/s137 runs are the wide arm (window 192, the dataset default).
# This is the narrow arm. Re10KDataset draws a random contiguous run of up to `window`
# frames and needs num_views*3 = 45 as its floor, so 48 is the narrowest legal setting.
set -u
GPUS=${1:-"0 1 2 3"}; WIN=${2:-48}; SEED=${3:-95}
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
[ "$WIN" -ge 45 ] || { echo "FATAL: window must be >= num_views*3 = 45"; exit 1; }

declare -A CFG=([base]=config/lact_l6_d256_p16.yaml [input]=config/cam_pra_hi.yaml
                [hidden]=config/cam_h_pra_hi.yaml [both]=config/cam_pra_h_hi.yaml)
ARMS=(base input hidden both); read -r -a G <<< "$GPUS"

run_one() {
  local GPU=$1 ARM=$2
  # separate statement: a single `local` expands all its words BEFORE assigning, so
  # referencing ARM in the same command hits `set -u` as unbound
  local EXP="dl3dvw${WIN}_${ARM}_s${SEED}"
  [ -f "outputs/$EXP/eval.json" ] && { echo "[$ARM] done"; return; }
  mkdir -p "outputs/$EXP"
  CUDA_VISIBLE_DEVICES=$GPU $PY -m torch.distributed.run \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
    train.py --config "${CFG[$ARM]}" \
    --data_path /tmp/dl3dv/train_index.json --dataset re10k --scene_pose_normalize \
    --window "$WIN" --expname "$EXP" \
    --steps 30000 --warmup 1500 --lr 1e-4 --lpips_start 5000 --seed "$SEED" \
    --bs_per_gpu 16 --num_all_views 15 --num_input_views 8 --num_target_views 8 \
    --image_size 256 256 --num_workers 7 --save_every 10000 --log_every 200 \
    > "outputs/$EXP/train.log" 2>&1
  echo "EXIT $? $EXP" >> outputs/exp_status.log
  # evaluate at the SAME narrow geometry it trained on, and at the wide default, so the
  # train-vs-test baseline mismatch is separable from the training effect itself
  for W in "$WIN" 128; do
    CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$EXP/model_0030000.pth" \
      --config "${CFG[$ARM]}" --data_path /tmp/dl3dv/test_index.json --num_scenes 140 \
      --window "$W" --min_frames 36 --out "outputs/$EXP/eval_w${W}.json" \
      > "outputs/$EXP/eval_w${W}.log" 2>&1
  done
  cp "outputs/$EXP/eval_w${WIN}.json" "outputs/$EXP/eval.json"
  echo "[$ARM] done"
}
for i in "${!ARMS[@]}"; do run_one "${G[$i]}" "${ARMS[$i]}" & sleep 5; done
wait
echo "[dl3dv window grid] all four arms done (window=$WIN seed=$SEED)"
