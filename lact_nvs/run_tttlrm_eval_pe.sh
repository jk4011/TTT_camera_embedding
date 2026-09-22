#!/bin/bash
# tab:recon for the PE baselines. PRoPE is evaluated at once (its cell is done); RayRoPE when
# its step-15000 checkpoint lands. Both CO-RESIDE on a card that is training (ALLOW_SHARED_GPU:
# an eval is ~10 GB next to a ~45-100 GB trainer), because the queue hands the freed cards to
# the ccv baselines immediately. The last call lists all four cells: the ladder skips the ones
# already evaluated and aggregates the four together, so the summary is over common scenes.
main() {
  cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/tttlrm_ref
  local LOG=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs/outputs/queue_tttlrm.log
  echo "$(date '+%F %T') tab:recon PE eval: prope on gpu3 (shared)" >> $LOG
  ALLOW_SHARED_GPU=1 CELLS="prope" STEPS=15000 GPU=3 bash eval_scratch_ladder.sh >> $LOG 2>&1
  until [ -f outputs/scratch_rayrope/step15000.pt ] || [ -f outputs/scratch_rayrope/final.pt ]; do sleep 60; done
  echo "$(date '+%F %T') tab:recon PE eval: rayrope + four-cell summary" >> $LOG
  ALLOW_SHARED_GPU=1 CELLS="base_re capet prope rayrope" STEPS=15000 GPU=3 bash eval_scratch_ladder.sh >> $LOG 2>&1
  echo "$(date '+%F %T') tab:recon PE eval finished (rc=$?)" >> $LOG
}
main "$@"
