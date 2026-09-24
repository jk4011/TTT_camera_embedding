#!/bin/bash
# table:ablation_design, Extrinsic and Projection rows re-run with ALL THREE placements
# (user, 2026-09-24): the 2026-09-18 cells used mat_in+h_mat only, while every rotary row in
# that table also carries the v/o transport, so the table was comparing 2 sites against 3.
# `mat_vo` adds the matrix analogue of that carrier (v <- M^-1 v, o <- M o), wired exactly as
# prope_orig wires its projective one. Same protocol as the rest of the table: seed 137,
# 8-view / 30k, RE10K 256x256 (256 eval scenes), gObjaverse orbit (499), DL3DV-u 256x448 (140).
main() {
  cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
  local G=${GPU:-0} LOG=outputs/queue_mat_vo.log
  for KIND in ext proj; do
    local CFG="config/mat_${KIND}_vo_both.yaml"
    echo "$(date '+%F %T') re10k_mat_${KIND}_vo_s137 start" >> $LOG
    NODE=node1 ./run_re10k.sh "$G" "re10k_mat_${KIND}_vo_s137" "$CFG" 137 >> $LOG 2>&1
    echo "$(date '+%F %T') gobj_mat_${KIND}_vo_s137 start" >> $LOG
    NODE=node1 DATA=gobj ./run_gobj.sh "$G" "gobj_mat_${KIND}_vo_s137" "$CFG" 137 >> $LOG 2>&1
    echo "$(date '+%F %T') dl3dvu_mat_${KIND}_vo_s137 start" >> $LOG
    NODE=node1 IMG="256 448" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      ./run_dl3dv.sh "$G" "dl3dvu_mat_${KIND}_vo_s137" "$CFG" 137 >> $LOG 2>&1
  done
  echo "$(date '+%F %T') all six matrix+carrier cells done" >> $LOG
}
main "$@"
