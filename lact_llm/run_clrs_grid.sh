#!/usr/bin/env bash
# Q29: CLRS-Text address-dimensionality grid.
#
# Usage:  ./run_clrs_grid.sh <gpu_list_csv> [token_budget]
#   e.g.  ./run_clrs_grid.sh 0,1,2,3            # node3, 4 free GPUs
#         ./run_clrs_grid.sh 0,1,2,3 400000000  # explicit budget
#
# The grid is 7 cells = {base} + {in, h, both} x {2d, 1d} coordinate mode.
# `base` has no rotary at all, so its coordinate is provably inert
# (sanity_clrs_coords.py mode b: |d| = 0.000e+00) -- it is ONE run, not two.
#
# The load-bearing comparison is, within each rotary arm, 2d vs 1d:
#   1d feeds (t, t), which recombines the split frequency ladder into
#   inv_freq * t, i.e. EXACTLY the stock rotary (verified bit-identical,
#   sanity_clrs_coords.py mode a). So 2d-vs-1d changes the address dimension
#   and nothing else -- same data, same tokens, same length, same memory load.
#
# Pre-registered prediction (write it down BEFORE reading any number):
#   the h-over-in increment appears in the 2d arms and vanishes in the 1d arms.
#   If h beats in equally in both, the dimensionality hypothesis (F20) is
#   refuted and this experiment says so.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <gpu_list_csv> [token_budget]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_TOP="$(dirname "$SCRIPT_DIR")"

IFS=',' read -r -a GPUS <<< "$1"
# The 2d_long corpus is 90,000 blocks x 4096 = 368.6M tokens/epoch, but only
# ~3-11% of each sequence is SUPERVISED (the answer region; the question is
# context). 1.2B tokens ~= 3.3 epochs ~= 90M supervised tokens, which is the
# same order as the F35 copy diagnostic's budget.
BUDGET="${2:-1200000000}"

# Locks live on lustre and are therefore SHARED BETWEEN NODES; scope them by
# hostname or node3's gpu0 will appear to collide with this node's gpu0.
LOCK_DIR="$REPO_TOP/lact_nvs/outputs/.gpu_locks"
HOST="$(hostname -s)"
mkdir -p "$LOCK_DIR"

# arm -> extra_json (ttt_nope / ttt_hidden_rope select the four rotary arms,
# matching the ledger's nope / rope / honly / hpra naming)
declare -A ARM_JSON=(
  [base]='{"ttt_nope": true}'
  [in]='{"ttt_nope": false}'
  [h]='{"ttt_nope": true, "ttt_hidden_rope": true, "ttt_hrope_gain": 1.0}'
  [both]='{"ttt_nope": false, "ttt_hidden_rope": true, "ttt_hrope_gain": 1.0}'
)

# cell name -> "arm coord_mode"
CELLS=(
  "q29_base      base 1d"
  "q29_in_2d     in   2d"
  "q29_in_1d     in   1d"
  "q29_h_2d      h    2d"
  "q29_h_1d      h    1d"
  "q29_both_2d   both 2d"
  "q29_both_1d   both 1d"
)

COMMON=(
  --data clrs --clrs_quadrant 2d_long
  --seq_len 4096 --window_size 128 --lact_chunk_size 1024
  --hidden_size 768 --num_hidden_layers 12 --num_attn_heads 12 --num_lact_heads 4
  --bs 8 --grad_accum 1 --lr 3e-4
  --token_budget "$BUDGET"
  --data_seed 42 --seed 42
  --val_every 1000 --log_every 100 --val_bs 8 --val_tokens 2000000
)

echo "[q29] host=$HOST gpus=${GPUS[*]} budget=$BUDGET cells=${#CELLS[@]}"

# Stripe cells over the GPU list; each GPU runs its cells sequentially.
for i in "${!GPUS[@]}"; do
  (
    g="${GPUS[$i]}"
    lock="$LOCK_DIR/${HOST}_gpu${g}"
    echo "q29-clrs" > "$lock"
    trap 'rm -f "$lock"' EXIT
    for ((j = i; j < ${#CELLS[@]}; j += ${#GPUS[@]})); do
      read -r name arm mode <<< "${CELLS[$j]}"
      out="outputs/$name"
      if [ -f "$out/val_log.jsonl" ] && grep -q '"step"' "$out/val_log.jsonl" 2>/dev/null \
         && [ -f "$out/final.pt" ]; then
        echo "[q29] gpu$g: $name already complete -> skip"
        continue
      fi
      echo "[q29] gpu$g: launching $name (arm=$arm coord=$mode)"
      mkdir -p "$out"
      # resubmission = resume: train_small.py auto-resumes from its own ckpt
      ./run_llm.sh "$g" "$name" \
          "${COMMON[@]}" \
          --clrs_coord_mode "$mode" \
          --extra_json "${ARM_JSON[$arm]}" \
        || echo "[q29] gpu$g: $name FAILED (see $out/train.log)"
    done
  ) &
done
wait
echo "[q29] all cells done on $HOST"
