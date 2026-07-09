#!/bin/bash
# queue_daemon.sh — dynamic GPU scheduler. Re-reads BATCH_QUEUE.txt every POLL
# seconds, so lines appended while it runs (by the user or by a remote-control
# Claude session) are picked up and launched on the next free GPU — no job
# resubmission needed. Completed runs (eval.json) are skipped; a run in flight
# is marked by outputs/.running_<exp>; GPUs are claimed via outputs/.gpu_locks/.
# Assumes it is the ONLY daemon on the node (stale locks cleared at start).
set -u
cd "$(dirname "$0")"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
QUEUE=${1:-BATCH_QUEUE.txt}
POLL=60
LOCKS=outputs/.gpu_locks
mkdir -p outputs "$LOCKS"
rm -f "$LOCKS"/gpu* outputs/.running_* 2>/dev/null
echo "[daemon] $(date '+%F %T') start node=$(hostname) queue=$QUEUE"

gpu_free() { # no lock file and <1 GiB in use
  [ -e "$LOCKS/gpu$1" ] && return 1
  local mem
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1")
  [ "${mem:-99999}" -lt 1024 ]
}

run_job() { # <gpu> <variant> <seed>: train (if needed) then eval, then release
  local G=$1 V=$2 S=$3 EXP CFG
  EXP="${V}_s${S}"; CFG="config/cam_${V}.yaml"
  if [ ! -f "$CFG" ]; then
    echo "[daemon] gpu$G SKIP $EXP: $CFG missing"
  else
    echo "[daemon] $(date '+%F %T') gpu$G TRAIN $EXP"
    [ -f "outputs/$EXP/model_0030000.pth" ] || bash launch_exp.sh "$G" "$EXP" "$CFG" "$S"
    if [ -f "outputs/$EXP/model_0030000.pth" ]; then
      CUDA_VISIBLE_DEVICES=$G $PY eval.py --load "outputs/$EXP/model_0030000.pth" --config "$CFG" \
        > "outputs/$EXP/eval.log" 2>&1
      echo "[daemon] $(date '+%F %T') gpu$G DONE $EXP: $(grep -h PSNR outputs/$EXP/eval.log | tail -1)"
    else
      echo "[daemon] $(date '+%F %T') gpu$G FAIL $EXP (see outputs/$EXP/train.log)"
    fi
  fi
  rm -f "$LOCKS/gpu$G" "outputs/.running_$EXP"
}

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
while true; do
  while read -r V S _; do
    [[ -z "${V:-}" || "$V" == \#* || -z "${S:-}" ]] && continue
    EXP="${V}_s${S}"
    [ -f "outputs/$EXP/eval.json" ] && continue
    [ -e "outputs/.running_$EXP" ] && continue
    for ((g = 0; g < NGPU; g++)); do
      if gpu_free "$g"; then
        echo "$EXP" > "$LOCKS/gpu$g"
        touch "outputs/.running_$EXP"
        run_job "$g" "$V" "$S" &
        break
      fi
    done
  done < "$QUEUE"
  # human/Claude-readable status snapshot
  {
    echo "updated: $(date '+%F %T')  (daemon pid $$)"
    for ((g = 0; g < NGPU; g++)); do
      echo "  gpu$g: $(cat "$LOCKS/gpu$g" 2>/dev/null || echo idle)"
    done
  } > outputs/QUEUE_STATUS.txt
  sleep "$POLL"
done
