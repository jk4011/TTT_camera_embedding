#!/bin/bash
# batch_entry.sh — self-contained Slurm-batch entry point (submit via web GUI).
#
#   bash /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_nvs/batch_entry.sh
#
# Requires nothing from the interactive session: env is the lustre venv, data
# is reshared from the lustre RE10K source into this node's /tmp, and the job
# list comes from BATCH_QUEUE.txt (one "variant seed" per line, cam_ prefix
# implied; '#' comments). Completed runs (eval.json present) are SKIPPED, so
# resubmitting after a kill simply continues where the last job stopped.
# Everything durable (checkpoints, logs, eval.json) lands in outputs/ on lustre.
set -u
cd "$(dirname "$0")"
PY_ENV=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin
PY=$PY_ENV/python
RE10K_SRC=/NHNHOME/WORKSPACE/26msit001_A/V-LAB/Datasets/re10k
QUEUE=${1:-BATCH_QUEUE.txt}
STAMP=$(date '+%F_%H%M%S')
MLOG="outputs/batch_${STAMP}.log"
mkdir -p outputs
exec > >(tee -a "$MLOG") 2>&1
echo "[batch] $(date) node=$(hostname) queue=$QUEUE"

# ---- 0. sanity: env + GPUs ----
$PY -c "import torch; assert torch.cuda.is_available()" || { echo "[batch] FATAL: no CUDA"; exit 1; }
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "[batch] $NGPU GPUs visible"

# ---- 1. data: reshard RE10K into this node's /tmp if missing ----
if [ ! -f /tmp/re10k/train_index.json ]; then
  echo "[batch] resharding RE10K -> /tmp/re10k"
  mkdir -p /tmp/re10k
  $PY data_preprocess/reshard_re10k.py --src "$RE10K_SRC/test"  --odir /tmp/re10k/test  --index /tmp/re10k/test_index.json  --workers 32
  $PY data_preprocess/reshard_re10k.py --src "$RE10K_SRC/train" --odir /tmp/re10k/train --index /tmp/re10k/train_index.json --workers 32
fi
echo "[batch] data ready: $($PY -c "import json;print(len(json.load(open('/tmp/re10k/train_index.json'))))") train scenes"

# ---- 2. build worklist: skip runs whose eval.json already exists ----
JOBS=()
while read -r V S _; do
  [[ -z "${V:-}" || "$V" == \#* ]] && continue
  EXP="${V}_s${S}"
  if [ -f "outputs/$EXP/eval.json" ]; then echo "[batch] SKIP $EXP (eval.json exists)"; continue; fi
  JOBS+=("$V:$S")
done < "$QUEUE"
echo "[batch] ${#JOBS[@]} jobs to run: ${JOBS[*]:-none}"
[ ${#JOBS[@]} -eq 0 ] && { echo "[batch] nothing to do"; exit 0; }

# ---- 3. one worker per GPU, striped job assignment ----
worker() {
  local GPU=$1; shift
  for job in "$@"; do
    local V="${job%%:*}" S="${job##*:}" EXP CFG CKPT
    EXP="${V}_s${S}"; CFG="config/cam_${V}.yaml"
    if [ ! -f "outputs/$EXP/model_0030000.pth" ]; then
      echo "[batch gpu$GPU] $(date '+%T') TRAIN $EXP"
      bash launch_exp.sh "$GPU" "$EXP" "$CFG" "$S"
    fi
    CKPT="outputs/$EXP/model_0030000.pth"
    if [ -f "$CKPT" ]; then
      echo "[batch gpu$GPU] $(date '+%T') EVAL $EXP"
      CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "$CKPT" --config "$CFG" > "outputs/$EXP/eval.log" 2>&1
      echo "[batch gpu$GPU] $EXP -> $(grep -h 'PSNR' outputs/$EXP/eval.log | tail -1)"
    else
      echo "[batch gpu$GPU] ERROR $EXP: no checkpoint after train (see outputs/$EXP/train.log)"
    fi
  done
}
PIDS=()
for ((g=0; g<NGPU; g++)); do
  ARGS=(); for ((i=g; i<${#JOBS[@]}; i+=NGPU)); do ARGS+=("${JOBS[$i]}"); done
  [ ${#ARGS[@]} -eq 0 ] && continue
  worker "$g" "${ARGS[@]}" & PIDS+=($!)
done
wait "${PIDS[@]}"

# ---- 4. summary table ----
echo "[batch] ===== SUMMARY $(date) ====="
for job in "${JOBS[@]}"; do
  EXP="${job%%:*}_s${job##*:}"
  R="outputs/$EXP/eval.json"
  if [ -f "$R" ]; then
    $PY -c "import json;r=json.load(open('$R'));print(f\"  $EXP: PSNR {r['psnr']:.3f} +- {r['psnr_std_err']:.3f}  LPIPS {r['lpips']:.4f}\")"
  else
    echo "  $EXP: FAILED"
  fi
done
echo "[batch] DONE"
