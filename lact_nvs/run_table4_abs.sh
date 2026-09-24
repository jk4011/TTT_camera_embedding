#!/bin/bash
# Table 4 (table:ablation_position) with ABSOLUTE depth (dpt_abs): 9 cells on ONE GPU.
# Three dataset lanes run side by side (always 3 cells on the card); each lane trains its rows in order
#   input+hidden (no v/o) -> input only -> hidden only
# so a whole table row lands at a time. The 4th CaPET row (input+hidden+v/o) is Table 3's cell, trained
# on node1 as *_dpabs_pdir_vo_s137 -- NOT here.
# Idempotent: each launcher skips a cell whose eval.json exists and resumes from its checkpoints, so
# after a node reset just run this again.   Usage: GPU=0 NODE=node2 bash run_table4_abs.sh
main() {
  local GPU=${GPU:-0}
  export NODE=${NODE:-node2}
  cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
  local LOG=outputs/queue_table4_abs.log
  lane() {
    local d=$1 v exp cfg
    for v in both in h; do
      exp=${d}_dpabs_pdir_${v}_s137; cfg=config/dp_abs_pdir_${v}.yaml
      echo "$(date '+%F %T') [table4] $exp start" >> $LOG
      case $d in
        re10k)  ./run_re10k.sh $GPU $exp $cfg 137 >> outputs/$exp.launch.log 2>&1 ;;
        gobj)   DATA=gobj ./run_gobj.sh $GPU $exp $cfg 137 >> outputs/$exp.launch.log 2>&1 ;;
        dl3dvu) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IMG="256 448" \
                  ./run_dl3dv.sh $GPU $exp $cfg 137 >> outputs/$exp.launch.log 2>&1 ;;
      esac
      echo "$(date '+%F %T') [table4] $exp exited rc=$?: $(tail -1 outputs/$exp.launch.log)" >> $LOG
    done
  }
  lane re10k & sleep 30
  lane gobj & sleep 30
  lane dl3dvu &
  wait
  echo "$(date '+%F %T') [table4] all nine cells done" >> $LOG
}
main "$@"
