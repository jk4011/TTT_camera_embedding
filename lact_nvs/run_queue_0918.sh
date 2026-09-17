#!/bin/bash
# Sequential queue for the 2026-09-18 NO study (matrix-only embeddings, RE10K + orbit, seed 137).
# Launches the next pending cell on whichever GPU has none of OUR cells (lock file outputs/.gpu_locks/node1_gpu<i>,
# written by run_re10k.sh / run_gobj.sh / run_dl3dv.sh and removed on exit). GPUs are shared with another project,
# so every cell runs with --actckpt (37 GB instead of 57 GB). Finished cells (eval.json) are skipped = resumable.
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
export NODE=node1
QUEUE=(
  "re10k re10k_mat_proj_s137 config/mat_proj_both.yaml"
  "gobj  gobj_mat_proj_s137  config/mat_proj_both.yaml"
  "re10k re10k_mat_ext_s137  config/mat_ext_both.yaml"
  "gobj  gobj_mat_ext_s137   config/mat_ext_both.yaml"
  "re10k re10k_mat_rot_s137  config/mat_rot_both.yaml"
  "gobj  gobj_mat_rot_s137   config/mat_rot_both.yaml"
)
i=0
while [ $i -lt ${#QUEUE[@]} ]; do
  for g in 3 0 1 2; do
    [ $i -lt ${#QUEUE[@]} ] || break
    [ -f outputs/.gpu_locks/node1_gpu$g ] && continue
    set -- ${QUEUE[$i]}; ds=$1; exp=$2; cfg=$3
    if [ -f outputs/$exp/eval.json ]; then echo "$(date '+%F %T') skip $exp (done)" >> outputs/queue_0918.log; i=$((i+1)); continue; fi
    case $ds in
      re10k) EXTRA_ARGS=--actckpt setsid nohup ./run_re10k.sh $g $exp $cfg 137 > outputs/$exp.launch.log 2>&1 < /dev/null & ;;
      gobj)  EXTRA_ARGS=--actckpt DATA=gobj setsid nohup ./run_gobj.sh $g $exp $cfg 137 > outputs/$exp.launch.log 2>&1 < /dev/null & ;;
    esac
    echo "$(date '+%F %T') launched $exp on gpu$g" >> outputs/queue_0918.log
    i=$((i+1)); sleep 45
  done
  sleep 60
done
echo "$(date '+%F %T') queue drained (all cells launched)" >> outputs/queue_0918.log
