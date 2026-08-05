#!/bin/bash
# Node3 was terminated with Q30 unfinished (13 of 15 cells). Its work moves here.
#
# The tttLRM grid runs each cell on a GPU PAIR but the NVS 80k arm chained behind it
# needs only ONE, so the odd GPU of every pair comes free and would otherwise idle for
# the rest of the day. This watcher puts a Q30 worker on each odd GPU the moment its
# pair's cell finishes.
#
# Pairs: both=0,1  h=2,3  in=4,5  base=6,7  ->  odd GPUs 1, 3, 5, 7.
#
# Each worker pulls cells from the shared atomically-claimed pool in
# lact_llm/outputs/.q30_claims, so staggered start times are fine and no two workers
# can take the same cell. File conditions only -- no `pgrep -f`, which matches this
# script's own command line.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTT=../tttlrm_ref
declare -A ODD=( [both]=1 [h]=3 [in]=5 [base]=7 )
DONE=""
while :; do
  for C in base in h both; do
    case " $DONE " in *" $C "*) continue;; esac
    [ -f "$TTT/outputs/scratch_$C/final.pt" ] || continue
    G=${ODD[$C]}
    echo "[$(date +%H:%M)] tttLRM cell '$C' finished -> Q30 worker on GPU $G"
    ( DIMS="2 4 6" ./run_grid_diag.sh "$G" >> "outputs/q30_worker_gpu${G}.log" 2>&1 ) &
    DONE="$DONE $C"
  done
  [ "$(echo $DONE | wc -w)" -eq 4 ] && break
  sleep 300
done
echo "[$(date +%H:%M)] all four Q30 workers launched; waiting for them"
wait
echo "[$(date +%H:%M)] Q30 pool drained"
