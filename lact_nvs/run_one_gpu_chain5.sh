#!/bin/bash
# node1 single-GPU chain, v5 (2026-09-24 21:40, user decisions with ~30 h to the deadline):
#  * NVS is finished with the FINAL recipe (dpt_lin+dpt_abs): the three Table 3 cells (run_seq.sh, running).
#  * ccv: NO more training (the RayRoPE / PRoPE cells are abandoned at steps 9249 / 8249). The EXISTING
#    CaPET checkpoint (ccv_capet_re, the older t_c recipe) is evaluated: its generation resumes at 42/64,
#    then camera accuracy (CPU, in the background).
#  * then table:diverse_fast_weight's three CaPET cells (2/3/4-layer MLP) with the final recipe.
# tttLRM moves to the 2-GPU allocation (run_tttlrm_capet_lin.sh), so nothing here waits for it.
# Locks use NODE=node1m: the forked session's Table 5 cells (run_mat_vo_six.sh) write node1_gpu0 from ITS
# node, and run_ccv_gen.sh would otherwise block on that foreign lock for hours.
# Idempotent: rerun after a reset; every step skips what is already on disk.
main() {
  local R=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope
  local A=$R/lact_ar_video/outputs N=$R/lact_nvs/outputs
  local LOG=$N/queue_chain.log
  export NODE=node1m
  say() { echo "$(date '+%F %T') [chain5] $*" >> $LOG; }
  gen_n() { ls $A/eval/gen_$1_13999/*_gen.mp4 2>/dev/null | wc -l; }

  say "waiting for the three dpt_lin NVS cells (they hold the card)"
  for e in re10k gobj dl3dvu; do until [ -f $N/${e}_dplin_pdir_vo_s137/eval.json ]; do sleep 120; done; done
  local DELTA
  DELTA=$(python3 -c "
import json
d=[json.load(open('$N/%s_dplin_pdir_vo_s137/eval.json'%e))['psnr']-json.load(open('$N/%s_dpchan_pdir_vo_s137/eval.json'%e))['psnr'] for e in ('re10k','gobj','dl3dvu')]
print('%.4f %s' % (sum(d)/3, ' '.join('%+.3f'%x for x in d)))")
  say "dpt_lin - t_c PSNR: mean ${DELTA%% *} (re10k gobj dl3dvu: ${DELTA#* })"

  if [ "$(gen_n ccv_capet_re)" -lt 64 ]; then
    say "ccv CaPET (existing checkpoint) generation resumes at $(gen_n ccv_capet_re)/64"
    bash $R/lact_nvs/run_ccv_gen.sh 0 capet ccv_capet_re
  fi
  if [ "$(gen_n ccv_capet_re)" -ge 64 ]; then
    say "ccv CaPET camera accuracy (CPU, background)"
    WHICHS=gen nohup setsid bash $R/lact_nvs/run_ccv_campose.sh ccv_capet_re >/dev/null 2>&1 &
  else
    say "ccv CaPET generation stopped at $(gen_n ccv_capet_re)/64, moving on"
  fi

  say "table:diverse_fast_weight: three CaPET cells with the final recipe"
  GPU=0 SEQ_LOG=$N/queue_seq_node1.log bash $R/lact_nvs/run_seq.sh \
    re10k:re10k_fw_mlp2_capetlin_s137:config/fw_mlp2_capetlin.yaml \
    re10k:re10k_fw_fw3l_capetlin_s137:config/fw_fw3l_capetlin.yaml \
    re10k:re10k_fw_fw4l_capetlin_s137:config/fw_fw4l_capetlin.yaml
  say "chain5 finished"
}
main "$@"
