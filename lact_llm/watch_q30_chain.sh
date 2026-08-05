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
    if [ "$C" = base ]; then
      # Longest job on the first GPU to free. The Q31 nope repair is 3B tokens, about
      # 3.75x one Q30 cell, so starting it last would leave it running alone for hours
      # after everything else is done. It falls through to the Q30 pool when finished.
      echo "[$(date +%H:%M)] tttLRM cell '$C' finished -> Q31 nope repair on GPU $G, then Q30"
      ( ./run_q31_nope_rerun.sh "$G" >> "outputs/q31_nope_rerun_gpu${G}.log" 2>&1
        DIMS="2 4 6" ./run_grid_diag.sh "$G" >> "outputs/q30_worker_gpu${G}.log" 2>&1 ) &
    elif [ "$C" = in ]; then
      # camimg first: it is 1.6 h and it decides an allocation question that gates the
      # CCV and Group A follow-ups, whereas Q30 is 13 exploratory cells at 2.4 h each.
      echo "[$(date +%H:%M)] tttLRM cell '$C' finished -> NVS camimg on GPU $G, then Q30"
      ( cd ../lact_nvs && ./run_camimg.sh "$G" 95 >> outputs/camimg_s95.launch.log 2>&1
        cd ../lact_llm && DIMS="2 4 6" ./run_grid_diag.sh "$G" >> "outputs/q30_worker_gpu${G}.log" 2>&1 ) &
    else
      # h -> gpu3 and both -> gpu1 take the 3D-reconstruction view sweep FIRST: it is a
      # figure the paper needs, whereas Q30 is exploratory. Both workers pull from one
      # atomically-claimed pool of 4 arms x 5 view counts, then fall through to Q30.
      # The sweep worker waits for each arm's final.pt itself, so starting before the
      # grid has finished is fine.
      echo "[$(date +%H:%M)] tttLRM cell '$C' finished -> 3D view sweep on GPU $G, then Q30"
      ( cd ../tttlrm_ref && ./run_scratch_viewsweep.sh "$G" >> logs/sweep_driver_gpu${G}.log 2>&1
        if [ "'$C'" = both ]; then
          /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python make_fig1.py \
            >> logs/make_fig1.log 2>&1
        fi
        cd ../lact_llm && DIMS="2 4 6" ./run_grid_diag.sh "$G" >> "outputs/q30_worker_gpu${G}.log" 2>&1 ) &
    fi
    DONE="$DONE $C"
  done
  [ "$(echo $DONE | wc -w)" -eq 4 ] && break
  sleep 300
done
echo "[$(date +%H:%M)] all four Q30 workers launched; waiting for them"
wait
echo "[$(date +%H:%M)] Q30 pool drained"
