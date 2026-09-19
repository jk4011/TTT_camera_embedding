#!/bin/bash
# Sequential PE-baseline queue (user request 2026-09-20): PRoPE on 3D reconstruction, then
# PRoPE on ccv, then RayRoPE on 3D reconstruction, then RayRoPE on ccv.
#
# Jobs run IN ORDER, each starting as soon as it has the GPUs it needs, so the order is
# respected without leaving cards idle behind a job that wants more of them.
# Queue file: one job per line, "<n_gpus> <label> <command with {GPUS} placeholder>".
# Lines starting with # and blank lines are ignored; a job whose DONE marker exists is skipped,
# so the queue can be resubmitted after a node reset.
set -u
REPO=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope
LOCKS=$REPO/lact_nvs/outputs/.gpu_locks
LOG=$REPO/lact_nvs/outputs/queue_pe.log
STATE=$REPO/lact_nvs/outputs/queue_pe_done
QUEUE=${1:-$REPO/lact_nvs/PE_QUEUE.txt}
HOST=node1
mkdir -p "$STATE" "$LOCKS"

free_gpus() {
  local out=""
  for g in 0 1 2 3 4 5 6 7; do
    nvidia-smi -i $g >/dev/null 2>&1 || continue
    local mem
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)
    # a foreign process can sit on a card without one of our locks, so both must be clear
    if [ "${mem:-99999}" -le 2048 ] && [ ! -f "$LOCKS/${HOST}_gpu$g" ]; then out="$out $g"; fi
  done
  echo $out
}

while IFS= read -r line; do
  case "$line" in ''|\#*) continue;; esac
  N=$(echo "$line" | awk '{print $1}')
  LABEL=$(echo "$line" | awk '{print $2}')
  CMD=$(echo "$line" | cut -d' ' -f3-)
  if [ -f "$STATE/$LABEL" ]; then
    echo "$(date '+%F %T') [pe-queue] $LABEL already done, skipping" >> $LOG
    continue
  fi
  echo "$(date '+%F %T') [pe-queue] $LABEL waiting for $N gpu(s)" >> $LOG
  PICK=""
  while :; do
    AVAIL=$(free_gpus)
    CNT=$(echo $AVAIL | wc -w)
    if [ "$CNT" -ge "$N" ]; then PICK=$(echo $AVAIL | cut -d' ' -f1-$N | tr ' ' ','); break; fi
    sleep 120
  done
  for g in ${PICK//,/ }; do echo "pe:$LABEL" > "$LOCKS/${HOST}_gpu$g"; done
  echo "$(date '+%F %T') [pe-queue] $LABEL start on gpu(s) $PICK" >> $LOG
  # started in the background so the NEXT job can claim cards this one does not need:
  # jobs still START in queue order, they just do not serialise behind a 40 h neighbour
  (
    eval "${CMD//\{GPUS\}/$PICK}" >> $LOG 2>&1
    RC=$?
    for g in ${PICK//,/ }; do rm -f "$LOCKS/${HOST}_gpu$g"; done
    echo "$(date '+%F %T') [pe-queue] $LABEL exited rc=$RC" >> $LOG
    [ $RC -eq 0 ] && touch "$STATE/$LABEL"
  ) &
  sleep 60          # let the job actually take the card before the next job polls
done < "$QUEUE"
echo "$(date '+%F %T') [pe-queue] all jobs launched; waiting for them to finish" >> $LOG
wait
echo "$(date '+%F %T') [pe-queue] queue drained" >> $LOG
