#!/bin/bash
# table:ablation_position, the missing "v/o only" row (user, 2026-09-25): the carrier alone,
# with the same channel-predicted depth every other row of that table uses
# (cam_mode dpt_chan+vo_rope, vo_coords foot). Same protocol: seed 137, 8-view / 30k.
main() {
  cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
  local G=${GPU:-0} CFG=config/dp_chan_vo_only.yaml LOG=outputs/queue_voonly.log
  echo "$(date '+%F %T') re10k_dpchan_voonly_s137 start" >> $LOG
  NODE=node1 ./run_re10k.sh "$G" re10k_dpchan_voonly_s137 "$CFG" 137 >> $LOG 2>&1
  echo "$(date '+%F %T') gobj_dpchan_voonly_s137 start" >> $LOG
  NODE=node1 DATA=gobj ./run_gobj.sh "$G" gobj_dpchan_voonly_s137 "$CFG" 137 >> $LOG 2>&1
  echo "$(date '+%F %T') dl3dvu_dpchan_voonly_s137 start" >> $LOG
  NODE=node1 IMG="256 448" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./run_dl3dv.sh "$G" dl3dvu_dpchan_voonly_s137 "$CFG" 137 >> $LOG 2>&1
  echo "$(date '+%F %T') all three v/o-only cells done" >> $LOG
}
main "$@"
