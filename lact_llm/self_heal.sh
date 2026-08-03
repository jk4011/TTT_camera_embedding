#!/usr/bin/env bash
# Self-healing wrapper around run_llm.sh for long DNA runs.
# Usage: ./self_heal.sh <gpu> <exp> [train_small.py args...]
# On a non-zero exit (crash / NCCL / transient), it preserves the crash tail and
# re-runs the SAME command up to 8 times, 60 s apart; --auto_resume (default true)
# picks up from the newest ckpt_step*.pt in outputs/<exp>. Exit 0 => run finished.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

gpu="$1"; exp="$2"; shift 2
MAX=8
for i in $(seq 1 "$MAX"); do
    echo "[self_heal] $exp attempt $i/$MAX on gpu$gpu $(date '+%F %T')"
    ./run_llm.sh "$gpu" "$exp" "$@"
    code=$?
    if [ "$code" -eq 0 ]; then
        echo "[self_heal] $exp finished cleanly on attempt $i"
        exit 0
    fi
    echo "[self_heal] $exp attempt $i exited $code"
    # preserve the crash tail before the next attempt truncates train.log
    if [ -f "outputs/$exp/train.log" ]; then
        cp "outputs/$exp/train.log" "outputs/$exp/train.log.crash$i" 2>/dev/null || true
    fi
    if [ "$i" -lt "$MAX" ]; then
        echo "[self_heal] retrying in 60s (auto-resume)"; sleep 60
    fi
done
echo "[self_heal] $exp FAILED after $MAX attempts"
exit 1
