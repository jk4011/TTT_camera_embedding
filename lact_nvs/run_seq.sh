#!/bin/bash
# Run NVS cells ONE AT A TIME on one GPU. The small model already saturates a B200: three cells side by
# side gave no more total throughput (per-cell it/s drops ~3x) and needed --actckpt to fit once LPIPS
# joins at step 5000, which costs recompute on top. One cell at a time needs neither, and the first
# result lands after ~2 h instead of all of them together.
# Idempotent: each launcher skips a finished cell (eval.json) and resumes from its checkpoints, so after
# a reset rerun the same command.
#   Usage: GPU=0 NODE=<node> [SEQ_LOG=outputs/<name>.log] bash run_seq.sh <dataset>:<exp>:<config> ...
#   dataset = re10k | gobj | dl3dvu (DL3DV uncropped, 256x448)
main() {
  local GPU=${GPU:-0}
  export NODE=${NODE:-$(hostname -s)}
  unset EXTRA_ARGS                      # alone on the card: no activation checkpointing needed
  cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
  local LOG=${SEQ_LOG:-outputs/queue_seq.log} spec d exp cfg rc
  for spec in "$@"; do
    IFS=: read -r d exp cfg <<< "$spec"
    echo "$(date '+%F %T') [seq] $exp start (gpu$GPU, $cfg)" >> $LOG
    case $d in
      re10k)  ./run_re10k.sh $GPU $exp $cfg 137 >> outputs/$exp.launch.log 2>&1 ;;
      gobj)   DATA=gobj ./run_gobj.sh $GPU $exp $cfg 137 >> outputs/$exp.launch.log 2>&1 ;;
      dl3dvu) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IMG="256 448" \
                ./run_dl3dv.sh $GPU $exp $cfg 137 >> outputs/$exp.launch.log 2>&1 ;;
      *) echo "$(date '+%F %T') [seq] unknown dataset '$d' in '$spec'" >> $LOG; continue ;;
    esac
    rc=$?
    echo "$(date '+%F %T') [seq] $exp exited rc=$rc: $(tail -1 outputs/$exp.launch.log)" >> $LOG
  done
  echo "$(date '+%F %T') [seq] all ${#@} cells done" >> $LOG
}
main "$@"
