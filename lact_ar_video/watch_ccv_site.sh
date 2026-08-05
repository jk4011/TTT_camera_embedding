#!/usr/bin/env bash
# P5 guard rail. Two jobs the NODE2_PROMPT calls out explicitly:
#   1. confirm a checkpoint dir actually appears after the first save interval
#      (a run that silently never checkpoints looks healthy until Slurm kills it);
#   2. copy step 13999 aside the moment it exists, because keep_last_iter=1000
#      prunes it once the run moves past it, and F30's eval ladder needs that step.
# Exits once all three cells have both (or have died), so the harness notifies.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CELLS="in h both"
OUT=outputs
KEEP_STEP=013999
ARCHIVE=outputs/_keep_step13999

alive() {  # alive <exp>: any python holding this out_dir
    local exp="$1" p c
    for p in $(ps -o pid= -C python); do
        c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
        case "$c" in *"ccv_site_${exp}"*) return 0 ;; esac
    done
    return 1
}

mkdir -p "$ARCHIVE"
first_seen=""
for i in $(seq 1 1440); do          # 1440 * 5 min = 5 days
    pending=""
    for c in $CELLS; do
        d="$OUT/ccv_site_$c/seed_1"
        # 1. first checkpoint
        if [ -z "${first_seen##*:$c:*}" ]; then :; else
            if ls -d "$d"/checkpoint_model_* >/dev/null 2>&1; then
                echo "[ckpt-ok] ccv_site_$c first checkpoint: $(ls -d "$d"/checkpoint_model_* | tail -1)"
                first_seen="$first_seen:$c:"
            fi
        fi
        # 2. archive step 13999 before pruning can remove it
        src="$d/checkpoint_model_$KEEP_STEP"
        dst="$ARCHIVE/ccv_site_${c}_checkpoint_model_$KEEP_STEP"
        if [ -d "$src" ] && [ ! -d "$dst" ]; then
            cp -r "$src" "$dst" && echo "[archive] ccv_site_$c step $KEEP_STEP copied aside"
        fi
        [ -d "$dst" ] || { alive "$c" && pending="$pending $c"; }
    done
    [ -z "$pending" ] && break
    sleep 300
done

echo "===== CCV SITE WATCHER DONE $(date '+%F %T') ====="
for c in $CELLS; do
    d="$OUT/ccv_site_$c/seed_1"
    st=$(grep -oE "^step [0-9]+" "$OUT/ccv_site_$c.log" 2>/dev/null | tail -1)
    n=$(ls -d "$d"/checkpoint_model_* 2>/dev/null | wc -l)
    arch=$([ -d "$ARCHIVE/ccv_site_${c}_checkpoint_model_$KEEP_STEP" ] && echo YES || echo no)
    alive "$c" && live=RUNNING || live=STOPPED
    printf "ccv_site_%-5s %-8s last=%-12s ckpts=%s step13999_archived=%s\n" "$c" "$live" "${st:-none}" "$n" "$arch"
done
