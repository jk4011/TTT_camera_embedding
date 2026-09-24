#!/bin/bash
# table:ablation_design, Extrinsic and Projection rows on DL3DV-uncropped (the other two
# datasets were run on 2026-09-18, F90). Same protocol as every other cell in that table:
# 8-view / 30k / seed 137, uncropped 256x448, eval on the 140 held-out scenes.
main() {
  cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
  local LOG=outputs/queue_mat_dl3dv.log
  for CELL in ext proj; do
    echo "$(date '+%F %T') dl3dvu_mat_${CELL}_s137 start" >> $LOG
    IMG="256 448" NODE=node1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      ./run_dl3dv.sh "${GPU:-0}" "dl3dvu_mat_${CELL}_s137" "config/mat_${CELL}_both.yaml" 137 >> $LOG 2>&1
    echo "$(date '+%F %T') dl3dvu_mat_${CELL}_s137 exited rc=$?" >> $LOG
  done
  echo "$(date '+%F %T') both matrix cells done" >> $LOG
}
main "$@"
