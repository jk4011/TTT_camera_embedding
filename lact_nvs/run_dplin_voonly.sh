#!/bin/bash
# table:ablation_position, "v/o only" row at the CURRENT recipe (2026-09-25): the table moved to
# the linear-head absolute depth (dpt_lin+dpt_abs) on all three datasets, so the carrier-only cell
# uses it too. The point:ray split (1:1 on RE10K/DL3DV, 3:1 on Objaverse) acts only on the input
# and hidden sites, which this cell does not have: the model is bit-identical under either
# config (same tensors, same init), so dp_lin_vo_only.yaml serves all three.
# The earlier dp_chan v/o-only cells were the superseded recipe (re10k finished: 22.370).
main() {
  cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
  local G=${GPU:-0} CFG=config/dp_lin_vo_only.yaml LOG=outputs/queue_dplin_voonly.log
  echo "$(date '+%F %T') re10k_dplin_voonly_s137 start" >> $LOG
  NODE=node1 ./run_re10k.sh "$G" re10k_dplin_voonly_s137 "$CFG" 137 >> $LOG 2>&1
  echo "$(date '+%F %T') gobj_dplin_voonly_s137 start" >> $LOG
  NODE=node1 DATA=gobj ./run_gobj.sh "$G" gobj_dplin_voonly_s137 "$CFG" 137 >> $LOG 2>&1
  echo "$(date '+%F %T') dl3dvu_dplin_voonly_s137 start" >> $LOG
  NODE=node1 IMG="256 448" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./run_dl3dv.sh "$G" dl3dvu_dplin_voonly_s137 "$CFG" 137 >> $LOG 2>&1
  echo "$(date '+%F %T') all three dplin v/o-only cells done" >> $LOG
}
main "$@"
