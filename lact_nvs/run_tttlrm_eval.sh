#!/bin/bash
# tab:recon evaluation: waits for both tttLRM cells to write their final checkpoint, then scores
# each on the 140 held-out DL3DV scenes with the cell's own trained view count.
# GPU: the first of GPU_CANDIDATES that is actually free. The original version hardcoded GPU 0,
# which the ccv cells now occupy; eval_scratch_ladder.sh would have refused to start on it.
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/tttlrm_ref
LOG=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs/outputs/queue_tttlrm.log
LOCKS=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs/outputs/.gpu_locks
GPU_CANDIDATES=${GPU_CANDIDATES:-"2 3 1 0"}
for d in scratch_base_re scratch_capet; do
  until [ -f outputs/$d/step15000.pt ] || [ -f outputs/$d/final.pt ]; do sleep 120; done
  echo "$(date '+%F %T') $d finished training" >> $LOG
done
echo "$(date '+%F %T') both cells done; waiting for a free GPU for tab:recon" >> $LOG
PICK=""
while [ -z "$PICK" ]; do
  for g in $GPU_CANDIDATES; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)
    if [ "${MEM:-99999}" -le 1024 ] && [ ! -f "$LOCKS/node1_gpu$g" ]; then PICK=$g; break; fi
  done
  [ -z "$PICK" ] && sleep 120
done
echo "$(date '+%F %T') starting tab:recon evaluation on gpu$PICK" >> $LOG
CELLS="base_re capet" STEPS=15000 GPU=$PICK bash eval_scratch_ladder.sh >> $LOG 2>&1
echo "$(date '+%F %T') tab:recon evaluation finished (rc=$?)" >> $LOG
