#!/bin/bash
# One tttLRM cell on a named GPU pair (tab:recon). Usage: run_tttlrm_pair.sh <g1,g2> <config> <tag> [wait]
# With "wait" it first blocks until both GPUs are free of our NVS locks, so the second cell can be
# queued while the first one is already training.
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/tttlrm_ref
PAIR=$1; CFG=$2; TAG=$3; WAITFIRST=${4:-}
LOCKS=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs/outputs/.gpu_locks
LOG=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs/outputs/queue_tttlrm.log
TORCHRUN=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/torchrun
export TRITON_CACHE_DIR=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/.cache_triton_tttlrm
export TORCHINDUCTOR_CACHE_DIR=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/.cache_inductor_tttlrm
export TORCHINDUCTOR_COMPILE_THREADS=1
# gsplat JIT-builds its CUDA backend into ~/.cache/torch_extensions, which a node reset wipes;
# the first step then dies on `cannot import name 'csrc' from 'gsplat'`. Build once on lustre.
export TORCH_EXTENSIONS_DIR=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/.cache_torchext_tttlrm
if [ -n "$WAITFIRST" ]; then
  for g in ${PAIR//,/ }; do until [ ! -f $LOCKS/node1_gpu$g ]; do sleep 120; done; done
fi
for g in ${PAIR//,/ }; do echo "tttlrm_$TAG" > $LOCKS/node1_gpu$g; done
PORT=$((29700 + ${PAIR%%,*}))
echo "$(date '+%F %T') tttlrm $TAG start on gpus $PAIR (port $PORT)" >> $LOG
CUDA_VISIBLE_DEVICES=$PAIR $TORCHRUN --nproc_per_node=2 --master_port=$PORT \
  train_cam.py $CFG >> outputs/${TAG}.log 2>&1
RC=$?
echo "$(date '+%F %T') tttlrm $TAG exited rc=$RC" >> $LOG
for g in ${PAIR//,/ }; do rm -f $LOCKS/node1_gpu$g; done
# the training's rc, not the lock cleanup's: a crash was reaching the queue as rc=0 and
# being marked done, so a resubmitted queue would have skipped it forever
exit $RC
