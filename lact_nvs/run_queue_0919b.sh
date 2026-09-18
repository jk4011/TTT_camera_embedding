#!/bin/bash
# DL3DV columns of BOTH ablation tables (2026-09-19, user): the four cells that were not run --
# table:ablation_position rows "input only" / "hidden only", and table:ablation_design rows
# "Ray" / "Ray+Cam". Waits until the baseline queue (run_queue_0919.sh) has launched everything so
# the paper's item-1 cells keep priority, then fills all four GPUs in one wave.
# --actckpt + expandable_segments as in every other DL3DV cell.
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
export NODE=node1
LOG=outputs/queue_0919b.log
until grep -q "queue drained" outputs/queue_0919.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') baseline queue drained; starting the four DL3DV ablation cells" >> $LOG
QUEUE=(
  "dl3dvu_dpchan_pdir_in_s137 config/dp_chan_pdir_in.yaml"
  "dl3dvu_dpchan_pdir_h_s137  config/dp_chan_pdir_h.yaml"
  "dl3dvu_ray_vo_s137         config/ray_vo_both.yaml"
  "dl3dvu_raycam_vo_s137      config/raycam_vo_both.yaml"
)
i=0
while [ $i -lt ${#QUEUE[@]} ]; do
  for g in 0 1 2 3; do
    [ $i -lt ${#QUEUE[@]} ] || break
    [ -f outputs/.gpu_locks/node1_gpu$g ] && continue
    set -- ${QUEUE[$i]}; exp=$1; cfg=$2
    if [ -f outputs/$exp/eval.json ]; then echo "$(date '+%F %T') skip $exp (done)" >> $LOG; i=$((i+1)); continue; fi
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_ARGS=--actckpt IMG="256 448" \
      setsid nohup ./run_dl3dv.sh $g $exp $cfg 137 > outputs/$exp.launch.log 2>&1 < /dev/null &
    echo "$(date '+%F %T') launched $exp on gpu$g" >> $LOG
    i=$((i+1)); sleep 45
  done
  sleep 60
done
echo "$(date '+%F %T') queue drained (all four launched)" >> $LOG
