#!/bin/bash
# Single-GPU allocation (2026-09-24): everything left, in priority order, on gpu0.
# Every step is idempotent, so resubmitting this script after a reset skips what is done.
#   1. tttLRM CaPET (corrected focus) finishes -> 2. tab:recon eval of all four cells
#   3. ccv CaPET generation (22 of 64 pairs left) -> camera metrics on CPU in the background
#   4. ccv RayRoPE training -> 5. ccv PRoPE training -> 6./7. their generation (+ metrics)
main() {
  local R=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope
  local LOCKS=$R/lact_nvs/outputs/.gpu_locks LOG=$R/lact_nvs/outputs/queue_chain.log
  local T=$R/tttlrm_ref
  say() { echo "$(date '+%F %T') [chain] $*" >> $LOG; }
  ccv_done() { [ -d $R/lact_ar_video/outputs/$1/seed_1/checkpoint_model_013999 ]; }
  gen_done() { [ "$(ls $R/lact_ar_video/outputs/eval/gen_$1_13999/*_gen.mp4 2>/dev/null | wc -l)" -ge 64 ]; }

  say "waiting for scratch_capet_axis to finish"
  until [ -f $T/outputs/scratch_capet_axis/step15000.pt ] || [ -f $T/outputs/scratch_capet_axis/final.pt ]; do sleep 60; done
  until ! pgrep -f "train_cam.py configs/scratch_capet_axis" >/dev/null; do sleep 30; done
  rm -f $LOCKS/node1_gpu0

  say "tab:recon eval (four cells)"
  (cd $T && CELLS="base_re prope rayrope capet_axis" STEPS=15000 GPU=0 bash eval_scratch_ladder.sh >> $R/lact_nvs/outputs/queue_tttlrm.log 2>&1)

  if ! gen_done ccv_capet_re; then say "ccv capet generation"; bash $R/lact_nvs/run_ccv_gen.sh 0 capet ccv_capet_re; fi
  gen_done ccv_capet_re && WHICHS=gen nohup setsid bash $R/lact_nvs/run_ccv_campose.sh ccv_capet_re >/dev/null 2>&1 &

  for c in "rayrope ccv_rayrope_re" "prope ccv_prope_re"; do
    set -- $c
    if ! ccv_done $2; then say "ccv $2 training"; bash $R/lact_nvs/run_ccv_start.sh 0 $1 $2; fi
  done
  for c in "rayrope ccv_rayrope_re" "prope ccv_prope_re"; do
    set -- $c
    if ccv_done $2 && ! gen_done $2; then say "ccv $2 generation"; bash $R/lact_nvs/run_ccv_gen.sh 0 $1 $2; fi
    gen_done $2 && WHICHS=gen nohup setsid bash $R/lact_nvs/run_ccv_campose.sh $2 >/dev/null 2>&1 &
  done
  say "chain finished"
}
main "$@"
