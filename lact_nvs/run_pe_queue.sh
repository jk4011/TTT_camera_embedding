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
HOST=node1                 # the convention our launchers write
REALHOST=$(hostname -s)    # what eval_scratch_ladder.sh writes
mkdir -p "$STATE" "$LOCKS"
# Locks from dead allocations would otherwise block every card forever, and a lock written
# under the OTHER prefix is invisible to a check that assumes one of them -- which is how a
# tttLRM cell and the tab:recon evaluation both landed on gpu2 on 2026-09-20.
# A lock older than this container is from a dead allocation BY DEFINITION, whatever its
# prefix: our launchers always write `node1_`, so a prefix check alone kept every card blocked
# for 7 h after the 2026-09-21 reallocation. /proc/1's mtime is the container start.
BOOT=/proc/1
for f in "$LOCKS"/*_gpu*; do
  [ -e "$f" ] || continue
  pre=$(basename "$f"); pre=${pre%_gpu*}
  if { [ "$pre" != "$HOST" ] && [ "$pre" != "$REALHOST" ]; } || [ ! "$f" -nt "$BOOT" ]; then
    echo "$(date '+%F %T') [pe-queue] clearing stale lock $(basename "$f") ($(cat "$f" 2>/dev/null))" >> $LOG
    rm -f "$f"
  fi
done

free_gpus() {
  local out=""
  for g in 0 1 2 3 4 5 6 7; do
    nvidia-smi -i $g >/dev/null 2>&1 || continue
    local mem
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)
    # a foreign process can sit on a card without one of our locks, so both must be clear
    if [ "${mem:-99999}" -le 2048 ] && [ ! -f "$LOCKS/${HOST}_gpu$g" ] \
       && [ ! -f "$LOCKS/${REALHOST}_gpu$g" ]; then out="$out $g"; fi
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
  # a job still running from an earlier queue instance (e.g. after a reorder) is not re-run
  if [ -f "$STATE/$LABEL.running" ]; then
    echo "$(date '+%F %T') [pe-queue] $LABEL is running elsewhere, skipping" >> $LOG
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
  # re-check after the wait: the job may have finished in another instance meanwhile
  if [ -f "$STATE/$LABEL" ]; then
    echo "$(date '+%F %T') [pe-queue] $LABEL finished elsewhere while waiting, skipping" >> $LOG
    continue
  fi
  for g in ${PICK//,/ }; do echo "pe:$LABEL" > "$LOCKS/${HOST}_gpu$g"; done
  echo "$(date '+%F %T') [pe-queue] $LABEL start on gpu(s) $PICK" >> $LOG
  # started in the background so the NEXT job can claim cards this one does not need:
  # jobs still START in queue order, they just do not serialise behind a 40 h neighbour
  (
    touch "$STATE/$LABEL.running"
    export PE_QUEUE_OWNED=1
    eval "${CMD//\{GPUS\}/$PICK}" >> $LOG 2>&1
    RC=$?
    for g in ${PICK//,/ }; do rm -f "$LOCKS/${HOST}_gpu$g"; done
    echo "$(date '+%F %T') [pe-queue] $LABEL exited rc=$RC" >> $LOG
    [ $RC -eq 0 ] && touch "$STATE/$LABEL"
    rm -f "$STATE/$LABEL.running"
  ) &
  sleep 60          # let the job actually take the card before the next job polls
done < "$QUEUE"
echo "$(date '+%F %T') [pe-queue] all jobs launched; waiting for them to finish" >> $LOG
wait
echo "$(date '+%F %T') [pe-queue] queue drained" >> $LOG
