#!/bin/bash
# Re-evaluate the 2026-09-19 tttLRM CaPET checkpoint with the code path it was TRAINED with
# (eager rotary, TTTROPE_NO_TRITON=1; legacy focus via focus_legacy in its config), to see
# whether evaluating it with the later Triton kernels moved its tab:recon number (F97).
main() {
  cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/tttlrm_ref
  local LOG=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs/outputs/queue_tttlrm.log
  echo "$(date '+%F %T') capet eager re-eval on gpu3 (shared)" >> $LOG
  TTTROPE_NO_TRITON=1 ALLOW_SHARED_GPU=1 CELLS="base_re capet capetEager" STEPS=15000 GPU=3 \
    bash eval_scratch_ladder.sh >> $LOG 2>&1
  echo "$(date '+%F %T') capet eager re-eval finished (rc=$?)" >> $LOG
}
main "$@"
