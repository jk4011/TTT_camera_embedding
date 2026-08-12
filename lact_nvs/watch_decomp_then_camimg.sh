#!/bin/bash
# Wait for the gobj decomposition cells, print the verdict, then start the revised
# both-datasets candidate: camimg on gObjaverse (gpu5). gpu7's use is decided after
# the verdict (this script exits and the harness notifies).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
until [ -f outputs/gobj_prope_imgrope_s95/eval.json ] && [ -f outputs/gobj_prope_raw_s95/eval.json ]; do sleep 240; done
/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python - <<'PY'
import json
print("=== gObjaverse decomposition of prope_orig's +0.32 ===")
for n,p in [("base","outputs/gobj_base_s95/eval.json"),
            ("imgrope only","outputs/gobj_prope_imgrope_s95/eval.json"),
            ("projective only","outputs/gobj_prope_raw_s95/eval.json"),
            ("full prope_orig","outputs/gobj_prope_orig_s95/eval.json")]:
    d=json.load(open(p)); print(f"  {n:18s} {d['psnr']:.3f}  {d['lpips']:.4f}  (n={d['num_scenes']})")
print("F34 RE10K reference: imgrope +0.379, projective-only -0.294")
PY
# launch the candidate regardless of verdict (camimg is informative either way);
# PROTOCOL NOTE: gobj needs --min_frames 40, run_camimg.sh does not pass it -> inline
EXP=gobj_camimg_s95
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "outputs/$EXP"
CUDA_VISIBLE_DEVICES=5 $PY -m torch.distributed.run \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=1 \
  train.py --config config/cam_camimg.yaml \
  --data_path /tmp/gobj/train_index.json --dataset re10k --scene_pose_normalize --min_frames 40 \
  --expname "$EXP" \
  --steps 30000 --warmup 1500 --lr 1e-4 --lpips_start 5000 --seed 95 \
  --bs_per_gpu 16 --num_all_views 15 --num_input_views 8 --num_target_views 8 \
  --image_size 256 256 --num_workers 7 --save_every 10000 --log_every 200 \
  > "outputs/$EXP/train.log" 2>&1
CUDA_VISIBLE_DEVICES=5 $PY eval.py --load "outputs/$EXP/model_0030000.pth" \
  --config config/cam_camimg.yaml --data_path /tmp/gobj/test_index.json --num_scenes 500 \
  > "outputs/$EXP/eval.log" 2>&1
echo "[camimg gobj] eval exit=$?"
