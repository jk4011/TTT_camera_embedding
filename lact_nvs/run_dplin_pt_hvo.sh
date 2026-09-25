#!/bin/bash
# POINT-ONLY hidden + v/o (user, 2026-09-25): starts after the point-only input+hidden+v/o chain
# (run_dplin_pt_vo.sh) has finished and released the card. Same hidden budget as that chain
# (3 x 84 = 252 point pairs), same carrier, no input site. Three datasets, seed 137, 8-view / 30k.
main() {
  cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
  local G=${GPU:-0} CFG=config/dp_lin_pt_hvo.yaml LOG=outputs/queue_dplin_pt_hvo.log
  until grep -q "all three point-only cells done" outputs/queue_dplin_pt_vo.log 2>/dev/null; do sleep 120; done
  while [ -f outputs/.gpu_locks/node1_gpu$G ]; do sleep 30; done
  echo "$(date '+%F %T') re10k_dplin_pt_hvo_s137 start" >> $LOG
  NODE=node1 ./run_re10k.sh "$G" re10k_dplin_pt_hvo_s137 "$CFG" 137 >> $LOG 2>&1
  echo "$(date '+%F %T') gobj_dplin_pt_hvo_s137 start" >> $LOG
  NODE=node1 DATA=gobj ./run_gobj.sh "$G" gobj_dplin_pt_hvo_s137 "$CFG" 137 >> $LOG 2>&1
  echo "$(date '+%F %T') dl3dvu_dplin_pt_hvo_s137 start" >> $LOG
  NODE=node1 IMG="256 448" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./run_dl3dv.sh "$G" dl3dvu_dplin_pt_hvo_s137 "$CFG" 137 >> $LOG 2>&1
  echo "$(date '+%F %T') all three point-only hidden+v/o cells done" >> $LOG
}
main "$@"
