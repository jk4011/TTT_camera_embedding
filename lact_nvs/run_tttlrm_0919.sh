#!/bin/bash
# tttLRM cells for tab:recon (paper item 5): No Encoding and CaPET, trained from scratch on DL3DV.
# Waits until all four GPUs are free of our NVS cells, then runs the two cells on 2 GPUs each
# (the protocol the earlier scratch_* runs used: bs 4/GPU x 2 ranks, 15k steps, ~12 h).
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/tttlrm_ref
LOCKS=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs/outputs/.gpu_locks
LOG=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs/outputs/queue_tttlrm.log
TORCHRUN=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/torchrun
export TRITON_CACHE_DIR=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/.cache_triton_tttlrm
export TORCHINDUCTOR_CACHE_DIR=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/.cache_inductor_tttlrm
export TORCHINDUCTOR_COMPILE_THREADS=1
until [ "$(ls $LOCKS 2>/dev/null | grep -c node1)" -eq 0 ]; do sleep 180; done
echo "$(date '+%F %T') all NVS cells done; launching the two tttLRM cells" >> $LOG
for g in 0 1 2 3; do echo "tttlrm" > $LOCKS/node1_gpu$g; done
( CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 --master_port=29700 \
    train_cam.py configs/scratch_base.yaml >> outputs/scratch_base_re.log 2>&1
  echo "$(date '+%F %T') tttlrm No Encoding exited rc=$?" >> $LOG
  rm -f $LOCKS/node1_gpu0 $LOCKS/node1_gpu1 ) &
sleep 30
( CUDA_VISIBLE_DEVICES=2,3 $TORCHRUN --nproc_per_node=2 --master_port=29702 \
    train_cam.py configs/scratch_capet.yaml >> outputs/scratch_capet.log 2>&1
  echo "$(date '+%F %T') tttlrm CaPET exited rc=$?" >> $LOG
  rm -f $LOCKS/node1_gpu2 $LOCKS/node1_gpu3 ) &
wait
echo "$(date '+%F %T') both tttLRM cells finished" >> $LOG
