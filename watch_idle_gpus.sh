#!/bin/bash
# Standing idle-GPU detector. EXITS when GPUs have been free for two consecutive
# polls, which makes the harness notify Claude so it can queue the next work.
#
# WHY THIS EXISTS. Every launcher in this repo is a FIXED chain: run this list, then
# stop. run_budget80k -> run_groupA, run_grid_diag over 15 cells, the view sweep over
# 20. Each one ends correctly and then nothing watches the machine. On 2026-08-06 the
# last chains drained at 10:23 and 10:38 and the node would have sat idle indefinitely
# if the user had not asked for status at 10:41. The gap was minutes that time; nothing
# in the setup bounded it.
#
# It deliberately does NOT auto-launch. Choosing the next experiment is a judgement
# call -- Q30 d=3/5 only made sense once the d=2/4/6 shape was visible, and the camimg
# seeds only because F46 landed below the noise floor. Waking Claude is the useful part.
#
# Two consecutive polls, because chained jobs leave a real gap between train and eval.
set -u
FREE_MIN=${FREE_MIN:-1}          # notify once this many GPUs are idle
POLL=${POLL:-300}
streak=0
while :; do
  idle=0; which=""
  for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g")
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "${n:-0}" -eq 0 ] && [ "${mem:-0}" -lt 1024 ]; then idle=$((idle+1)); which="$which $g"; fi
  done
  if [ "$idle" -ge "$FREE_MIN" ]; then
    streak=$((streak+1))
    echo "[$(date +%H:%M)] $idle idle GPU(s):$which  (streak $streak)"
    if [ "$streak" -ge 2 ]; then
      echo "IDLE GPUS:$which -- queue drained, Claude should pick the next experiment"
      exit 0
    fi
  else
    streak=0
  fi
  sleep "$POLL"
done
