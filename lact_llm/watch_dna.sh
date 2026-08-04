#!/usr/bin/env bash
# Watcher for the four DNA runs: exits (notifying Claude) once every run has
# either finished (final.pt written) or died (no live python for its out_dir).
# File/process-state based on purpose — no `pgrep -f`, which self-matches.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Runs to watch: pass as args, else default to the WAVE 1 grid.
if [ "$#" -gt 0 ]; then
    RUNS="$*"
else
    RUNS="dna_nope_w128 dna_rope_w128 dna_honly_g1_w128 dna_hpra_g1_w128"
fi

alive() {  # alive <exp> -> 0 if a python proc has this out_dir in its cmdline
    local exp="$1" p c
    for p in $(ps -o pid= -C python); do
        c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
        case "$c" in *"outputs/$exp"*) return 0 ;; esac
    done
    return 1
}

for i in $(seq 1 288); do          # 288 * 5 min = 24 h ceiling
    pending=""
    for r in $RUNS; do
        if [ -f "outputs/$r/final.pt" ]; then continue; fi   # finished
        if alive "$r"; then pending="$pending $r"; fi        # still training
        # neither: dead without final.pt -> treat as settled (failure), report below
    done
    [ -z "$pending" ] && break
    sleep 300
done

echo "===== WATCHER DONE $(date '+%F %T') ====="
for r in $RUNS; do
    if [ -f "outputs/$r/final.pt" ]; then st="FINISHED"
    elif alive "$r"; then st="STILL-RUNNING(timeout)"
    else st="DEAD-NO-FINAL"; fi
    printf "%-22s %s  " "$r" "$st"
    tail -1 "outputs/$r/val_log.jsonl" 2>/dev/null || echo "(no val_log)"
done
