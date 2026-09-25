#!/bin/bash
# POINT-ONLY CaPET at all three placements (user, 2026-09-25): the full recipe with the ray-direction
# half removed and its phase budget given to the point (input 3 x 42 = 126 pairs, hidden 3 x 84 = 252,
# i.e. the same totals the 1:1 and 3:1 split cells use). Linear-head absolute depth as the current
# recipe; seed 137, 8-view / 30k on all three datasets. One config: with no ray half there is no split.
main() {
  cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
  local G=${GPU:-0} CFG=config/dp_lin_pt_vo_both.yaml LOG=outputs/queue_dplin_pt_vo.log
  echo "$(date '+%F %T') re10k_dplin_pt_vo_s137 start" >> $LOG
  NODE=node1 ./run_re10k.sh "$G" re10k_dplin_pt_vo_s137 "$CFG" 137 >> $LOG 2>&1
  echo "$(date '+%F %T') gobj_dplin_pt_vo_s137 start" >> $LOG
  NODE=node1 DATA=gobj ./run_gobj.sh "$G" gobj_dplin_pt_vo_s137 "$CFG" 137 >> $LOG 2>&1
  echo "$(date '+%F %T') dl3dvu_dplin_pt_vo_s137 start" >> $LOG
  NODE=node1 IMG="256 448" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./run_dl3dv.sh "$G" dl3dvu_dplin_pt_vo_s137 "$CFG" 137 >> $LOG 2>&1
  echo "$(date '+%F %T') all three point-only cells done" >> $LOG
}
main "$@"
