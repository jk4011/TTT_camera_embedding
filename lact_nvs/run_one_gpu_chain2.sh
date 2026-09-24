#!/bin/bash
# node1 single-GPU chain, v2 (2026-09-24, user decision): once the three NVS dpt_abs cells are in,
# RETRAIN the ccv CaPET cell with ABSOLUTE depth -- unless NVS says absolute depth is clearly worse
# (mean PSNR over the three datasets more than 0.15 dB below the t_c recipe), in which case the old
# plan resumes (t_c CaPET generation). Then the ccv baselines. Every step is idempotent (resubmit after
# a reset); a ccv training another node has CLAIMED is skipped. Completion is read from files only.
main() {
  local R=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope
  local T=$R/tttlrm_ref A=$R/lact_ar_video/outputs N=$R/lact_nvs/outputs
  local LOG=$N/queue_chain.log ME=${NODE:-node1}
  say() { echo "$(date '+%F %T') [chain2] $*" >> $LOG; }
  ccv_done() { [ -d $A/$1/seed_1/checkpoint_model_013999 ]; }
  gen_done() { [ "$(ls $A/eval/gen_$1_13999/*_gen.mp4 2>/dev/null | wc -l)" -ge 64 ]; }
  claimed_elsewhere() { [ -f $A/$1.claimed ] && [ "$(cat $A/$1.claimed)" != "$ME" ]; }
  train() {   # <variant> <exp>
    if ccv_done $2; then return 0; fi
    if claimed_elsewhere $2; then say "$2 claimed by $(cat $A/$2.claimed), skipping"; return 0; fi
    echo "$ME" > $A/$2.claimed
    say "ccv $2 training"
    PE_QUEUE_OWNED=1 bash $R/lact_nvs/run_ccv_start.sh 0 $1 $2
    ccv_done $2 || say "ccv $2 FAILED (no step-13999 checkpoint), moving on"
  }
  gen() {     # <variant> <exp>
    ccv_done $2 || return 0
    if ! gen_done $2; then say "ccv $2 generation"; bash $R/lact_nvs/run_ccv_gen.sh 0 $1 $2; fi
    gen_done $2 && WHICHS=gen nohup setsid bash $R/lact_nvs/run_ccv_campose.sh $2 >/dev/null 2>&1 &
  }

  say "waiting for the tab:recon eval (capet_axis) and the three dpt_abs NVS cells"
  until [ "$(ls $T/evaluation/scratch_capet_axis_step15000 2>/dev/null | wc -l)" -ge 140 ]; do sleep 60; done
  for e in re10k gobj dl3dvu; do until [ -f $N/${e}_dpabs_pdir_vo_s137/eval.json ]; do sleep 120; done; done

  local DELTA
  DELTA=$(python3 -c "
import json
d=[json.load(open('$N/%s_dpabs_pdir_vo_s137/eval.json'%e))['psnr']-json.load(open('$N/%s_dpchan_pdir_vo_s137/eval.json'%e))['psnr'] for e in ('re10k','gobj','dl3dvu')]
print('%.4f %s' % (sum(d)/3, ' '.join('%+.3f'%x for x in d)))")
  say "dpt_abs - t_c PSNR: mean ${DELTA%% *} (re10k gobj dl3dvu: ${DELTA#* })"
  if python3 -c "import sys; sys.exit(0 if float('${DELTA%% *}') >= -0.15 else 1)"; then
    say "branch ABS: retraining ccv CaPET with absolute depth"
    train capet_abs ccv_capetabs_re
    gen capet_abs ccv_capetabs_re
  else
    say "branch T_C: absolute depth is > 0.15 dB worse; resuming the t_c CaPET generation"
    gen capet ccv_capet_re
  fi

  for c in "rayrope ccv_rayrope_re" "prope ccv_prope_re"; do set -- $c; train $1 $2; done
  for c in "rayrope ccv_rayrope_re" "prope ccv_prope_re"; do set -- $c; gen $1 $2; done
  say "chain2 finished"
}
main "$@"
