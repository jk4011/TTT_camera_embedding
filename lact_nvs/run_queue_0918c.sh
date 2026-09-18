#!/bin/bash
# Sequential queue for the final-recipe ablation (2026-09-18 21:50): input-only / hidden-only (no v/o), 3 datasets, seed 137.
# Same lock-file logic as run_queue_0918.sh; every cell --actckpt (identical setting to the final-recipe rows).
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
export NODE=node1
QUEUE=(
  "re10k  re10k_dpchan_pdir_in_s137  config/dp_chan_pdir_in.yaml"
  "gobj   gobj_dpchan_pdir_in_s137   config/dp_chan_pdir_in.yaml"
  "dl3dvu dl3dvu_dpchan_pdir_in_s137 config/dp_chan_pdir_in.yaml"
  "re10k  re10k_dpchan_pdir_h_s137   config/dp_chan_pdir_h.yaml"
  "gobj   gobj_dpchan_pdir_h_s137    config/dp_chan_pdir_h.yaml"
  "dl3dvu dl3dvu_dpchan_pdir_h_s137  config/dp_chan_pdir_h.yaml"
)
LOG=outputs/queue_0918c.log
i=0
while [ $i -lt ${#QUEUE[@]} ]; do
  for g in 0 1 2 3; do
    [ $i -lt ${#QUEUE[@]} ] || break
    [ -f outputs/.gpu_locks/node1_gpu$g ] && continue
    set -- ${QUEUE[$i]}; ds=$1; exp=$2; cfg=$3
    if [ -f outputs/$exp/eval.json ]; then echo "$(date '+%F %T') skip $exp (done)" >> $LOG; i=$((i+1)); continue; fi
    case $ds in
      re10k)  EXTRA_ARGS=--actckpt setsid nohup ./run_re10k.sh $g $exp $cfg 137 > outputs/$exp.launch.log 2>&1 < /dev/null & ;;
      gobj)   EXTRA_ARGS=--actckpt DATA=gobj setsid nohup ./run_gobj.sh $g $exp $cfg 137 > outputs/$exp.launch.log 2>&1 < /dev/null & ;;
      dl3dvu) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_ARGS=--actckpt IMG="256 448" setsid nohup ./run_dl3dv.sh $g $exp $cfg 137 > outputs/$exp.launch.log 2>&1 < /dev/null & ;;
    esac
    echo "$(date '+%F %T') launched $exp on gpu$g" >> $LOG
    i=$((i+1)); sleep 45
  done
  sleep 60
done
echo "$(date '+%F %T') queue drained (all cells launched)" >> $LOG
