#!/bin/bash
# Re-queue of dl3dvu_prope_s137 (the main queue had already advanced past it when it died on the
# square-patch-grid assertion, now fixed). Takes the first GPU that frees.
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
export NODE=node1
LOG=outputs/queue_0919.log
exp=dl3dvu_prope_s137; cfg=config/cam_prope_orig.yaml
while true; do
  [ -f outputs/$exp/eval.json ] && { echo "$(date '+%F %T') $exp already done" >> $LOG; exit 0; }
  if pgrep -f "run_[a-z0-9]*\.sh [0-9] $exp " > /dev/null 2>&1; then sleep 120; continue; fi
  for g in 0 1 2 3; do
    [ -f outputs/.gpu_locks/node1_gpu$g ] && continue
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IMG="256 448" \
      setsid nohup ./run_dl3dv.sh $g $exp $cfg 137 > outputs/$exp.launch.log 2>&1 < /dev/null &
    echo "$(date '+%F %T') launched $exp on gpu$g (requeue)" >> $LOG
    exit 0
  done
  sleep 60
done
