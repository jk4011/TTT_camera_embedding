#!/bin/bash
# fig:input-scale sweeps (paper item 2). Each dataset starts as soon as its three baseline
# checkpoints exist; evals are small enough to share a GPU with a training cell, so this does not
# wait for the training queues to drain. Runs on whichever GPU currently holds the least memory.
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
LOG=outputs/sweeps_0919.log
pick_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | tr -d ' ' | cut -d, -f1; }
run_ds() {
  local DS=$1; shift
  for ck in "$@"; do until [ -f "outputs/$ck/model_0030000.pth" ]; do sleep 180; done; done
  local G=$(pick_gpu)
  echo "$(date '+%F %T') $DS sweep start on gpu$G" >> $LOG
  ./run_vsweep_paper.sh $DS $G >> outputs/sweep_${DS}.log 2>&1
  echo "$(date '+%F %T') $DS sweep done (exit $?)" >> $LOG
}
run_ds re10k  base_s137_re re10k_prope_s137 re10k_gta_s137
run_ds gobj   gobj_base_s137_re gobj_prope_s137 gobj_gta_s137
run_ds dl3dvu dl3dvu_base_s137_re dl3dvu_prope_s137 dl3dvu_gta_s137
echo "$(date '+%F %T') ALL SWEEPS DONE" >> $LOG
