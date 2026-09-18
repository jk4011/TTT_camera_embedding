#!/bin/bash
# Baselines for the QUALITATIVE FIGURE (2026-09-19, user): NoPE / PRoPE / GTA retrained at seed 137 --
# their checkpoints were deleted in the 2026-09-07 purge. RayRoPE (sigma0=3) and our recipe still have theirs.
# RE10K + orbit; standard protocol; --actckpt not needed (GPUs are ours alone) but kept off for speed.
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
export NODE=node1
QUEUE=(
  "re10k base_s137_re      config/lact_l6_d256_p16.yaml"
  "gobj  gobj_base_s137_re config/lact_l6_d256_p16.yaml"
  "re10k re10k_prope_s137  config/cam_prope_orig.yaml"
  "gobj  gobj_prope_s137   config/cam_prope_orig.yaml"
  "re10k re10k_gta_s137    config/cam_gta_in.yaml"
  "gobj  gobj_gta_s137     config/cam_gta_in.yaml"
  "dl3dvu dl3dvu_base_s137_re config/lact_l6_d256_p16.yaml"
  "dl3dvu dl3dvu_prope_s137   config/cam_prope_orig.yaml"
  "dl3dvu dl3dvu_gta_s137     config/cam_gta_in.yaml"
)
LOG=outputs/queue_0919.log
i=0
while [ $i -lt ${#QUEUE[@]} ]; do
  for g in 0 1 2 3; do
    [ $i -lt ${#QUEUE[@]} ] || break
    [ -f outputs/.gpu_locks/node1_gpu$g ] && continue
    set -- ${QUEUE[$i]}; ds=$1; exp=$2; cfg=$3
    if [ -f outputs/$exp/eval.json ]; then echo "$(date '+%F %T') skip $exp (done)" >> $LOG; i=$((i+1)); continue; fi
    # never launch a cell that is already training (a restarted runner would otherwise
    # start a duplicate on a second GPU: both truncate the same log and race on the
    # checkpoint). The wrapper's command line carries the experiment name.
    if pgrep -f "run_[a-z0-9]*\.sh [0-9] $exp " > /dev/null 2>&1; then
      echo "$(date '+%F %T') skip $exp (already running)" >> $LOG; i=$((i+1)); continue; fi
    case $ds in
      re10k) setsid nohup ./run_re10k.sh $g $exp $cfg 137 > outputs/$exp.launch.log 2>&1 < /dev/null & ;;
      gobj)  DATA=gobj setsid nohup ./run_gobj.sh $g $exp $cfg 137 > outputs/$exp.launch.log 2>&1 < /dev/null & ;;
      dl3dvu) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IMG="256 448" setsid nohup ./run_dl3dv.sh $g $exp $cfg 137 > outputs/$exp.launch.log 2>&1 < /dev/null & ;;
    esac
    echo "$(date '+%F %T') launched $exp on gpu$g" >> $LOG
    i=$((i+1)); sleep 45
  done
  sleep 60
done
echo "$(date '+%F %T') queue drained (all cells launched)" >> $LOG
