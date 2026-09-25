#!/bin/bash
# Continuation of run_voonly_three.sh after the user moved Objaverse to the 3/4-point, 1/4-ray
# recipe (2026-09-25): RE10K and DL3DV keep dp_chan_vo_only.yaml (their table rows use the value-
# channel depth); the Objaverse v/o-only cell is launched separately once its config is settled.
main() {
  cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
  local G=${GPU:-0} LOG=outputs/queue_voonly.log
  until [ -f outputs/re10k_dpchan_voonly_s137/eval.json ]; do sleep 60; done
  while [ -f outputs/.gpu_locks/node1_gpu$G ]; do sleep 30; done      # let the re10k wrapper release
  echo "$(date '+%F %T') dl3dvu_dpchan_voonly_s137 start" >> $LOG
  NODE=node1 IMG="256 448" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./run_dl3dv.sh "$G" dl3dvu_dpchan_voonly_s137 config/dp_chan_vo_only.yaml 137 >> $LOG 2>&1
  echo "$(date '+%F %T') dl3dvu v/o-only done" >> $LOG
}
main "$@"
