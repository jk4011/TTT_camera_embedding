#!/bin/bash
# tab:recon point-only ablation: the final recipe WITHOUT the ray-direction half (pdir), whole budget on the point, 2026-09-25.
# Meant for the 2-GPU allocation: 2 GPUs x bs 4 at 2.7 s/step -> 15k steps in ~11.3 h. On a 1-GPU node it
# falls back to grad_accum 2 (same effective batch 8, 5.8 s/step, ~24 h). Both configs write the same
# checkpoint dir and train_cam.py auto-resumes from latest.pt, so the run can move between allocations.
# After training it scores step 15000 on the 140 held-out DL3DV scenes, like the other tab:recon cells.
# Usage (on the new node, from anywhere):
#   NODE=<name> nohup setsid bash /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs/run_tttlrm_capet_lin.sh >/dev/null 2>&1 &
main() {
  local R=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope
  cd $R/tttlrm_ref || exit 1
  local LOCKS=$R/lact_nvs/outputs/.gpu_locks LOG=$R/lact_nvs/outputs/queue_tttlrm.log
  local ME=${NODE:-$(hostname -s)} TAG=scratch_capet_pt
  local TORCHRUN=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/torchrun
  export TRITON_CACHE_DIR=$R/.cache_triton_tttlrm TORCHINDUCTOR_CACHE_DIR=$R/.cache_inductor_tttlrm
  export TORCHINDUCTOR_COMPILE_THREADS=1
  export TORCH_EXTENSIONS_DIR=$R/.cache_torchext_tttlrm   # gsplat's CUDA build, kept on lustre
  local NG CFG GPUS NP RC
  NG=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [ "$NG" -ge 2 ]; then CFG=configs/scratch_capet_pt.yaml; GPUS=0,1; NP=2
  else CFG=configs/scratch_capet_pt_1gpu.yaml; GPUS=0; NP=1; fi
  if [ ! -f outputs/$TAG/step15000.pt ] && [ ! -f outputs/$TAG/final.pt ]; then
    for g in ${GPUS//,/ }; do echo "tttlrm_$TAG" > $LOCKS/${ME}_gpu$g; done
    echo "$(date '+%F %T') tttlrm $TAG start on $ME gpus $GPUS ($CFG)" >> $LOG
    CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN --nproc_per_node=$NP --master_port=29713 \
      train_cam.py $CFG >> outputs/$TAG.log 2>&1
    RC=$?
    for g in ${GPUS//,/ }; do rm -f $LOCKS/${ME}_gpu$g; done
    echo "$(date '+%F %T') tttlrm $TAG exited rc=$RC" >> $LOG
    [ $RC -ne 0 ] && exit $RC
  fi
  echo "$(date '+%F %T') tttlrm $TAG: tab:recon evaluation" >> $LOG
  # ALLOW_SHARED_GPU=1: another project may hold a little memory on this card; without it the eval
  # refuses to start whenever more than 1 GB is in use (it still refuses below 30 GB free).
  ALLOW_SHARED_GPU=1 CELLS=capet_pt STEPS=15000 GPU=0 bash eval_scratch_ladder.sh >> $LOG 2>&1
  # the CELLS=capet_pt eval rewrites scratch_metrics_step15000.json with one cell; rebuild the full table
  local A="" c
  for c in base_re prope rayrope capet capet_axis capet_lin capet_pt; do A="$A $c=evaluation/scratch_${c}_step15000"; done
  /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python aggregate_metrics.py --eval_dirs $A --baseline base_re \
    --out scratch_metrics_step15000.json --csv scratch_per_scene_step15000.csv >> $LOG 2>&1
  echo "$(date '+%F %T') tttlrm $TAG: evaluation finished ($(ls evaluation/scratch_capet_pt_step15000 2>/dev/null | wc -l)/140 scenes)" >> $LOG
}
main "$@"
