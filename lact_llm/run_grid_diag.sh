#!/usr/bin/env bash
# Q30: grid-recall diagnostic with an ADDRESS-DIMENSION sweep.
#
# Usage:  ./run_grid_diag.sh <gpu_list_csv> [token_budget]
#   e.g.  ./run_grid_diag.sh 0,1,2,3
#         DIMS="1 2 5" ARMS="h" ./run_grid_diag.sh 0,1
#
# The stored tokens are IDENTICAL at every coord_dims -- byte for byte, verified
# by synthetic_grid.selftest(). Only the factorisation of the flat index into
# coordinate axes changes. So this measures address REPRESENTATION with the
# task, the retrieval pattern, the memory load and the answer length all held
# fixed.
#
# k=1 is the stock rotary (one band = the whole ladder), so the k=1 cells are
# the baseline that every larger k is compared against WITHIN its own arm.
#
# Pre-registered prediction: NOT a monotone gain. _ext_angles gives each axis
# only ~P/k frequencies, so extra axes buy structure and spend resolution. The
# curve should peak near the data's true structure (k=2 here: fiber position x
# fiber index) and decay after. The load-bearing comparison is whether the
# hidden site's curve peaks HIGHER or LATER than the input site's -- that is the
# claim that the hidden site is the multi-dimensional-address mechanism.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <gpu_list_csv> [token_budget]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_TOP="$(dirname "$SCRIPT_DIR")"

IFS=',' read -r -a GPUS <<< "$1"
# Sequences are generated on the fly and never repeat, so there is no epoch.
# 800M matches the F35 copy diagnostic this task supersedes.
BUDGET="${2:-800000000}"
DIMS=${DIMS:-"2 3 4 5 6"}
ARMS=${ARMS:-"in h"}

# Locks live on lustre and are therefore SHARED BETWEEN NODES; scope them by
# hostname or node3's gpu0 collides with node1's gpu0.
LOCK_DIR="$REPO_TOP/lact_nvs/outputs/.gpu_locks"
HOST="$(hostname -s)"
mkdir -p "$LOCK_DIR"

declare -A ARM_JSON=(
  [base]='{"ttt_nope": true}'
  [in]='{"ttt_nope": false}'
  [h]='{"ttt_nope": true, "ttt_hidden_rope": true, "ttt_hrope_gain": 1.0}'
  [both]='{"ttt_nope": false, "ttt_hidden_rope": true, "ttt_hrope_gain": 1.0}'
)

# base has no rotary, so its coordinate is provably inert
# (verify_clrs_coords_active asserts exactly 0.0 for it) -- one base per d, not
# per coord_mode. It is the FLOOR: it says how much harder the task itself gets
# as the lattice deepens, which the rotary arms must be read against.
CELLS=()
for k in $DIMS; do
  CELLS+=("q30_base_d${k} base $k 1d")     # no rotary: the floor at each d
  for arm in $ARMS; do
    CELLS+=("q30_${arm}_d${k}_nd $arm $k 2d")   # true d-D address
    CELLS+=("q30_${arm}_d${k}_flat $arm $k 1d") # flat address == stock rotary
  done
done

COMMON=(
  --data grid
  --seq_len 4096 --window_size 128 --lact_chunk_size 1024
  --hidden_size 768 --num_hidden_layers 12 --num_attn_heads 12 --num_lact_heads 4
  --bs 8 --grad_accum 1 --lr 3e-4
  --token_budget "$BUDGET"
  --data_seed 42 --seed 42
  --val_every 1000 --log_every 100 --val_bs 8
)

echo "[q30] host=$HOST gpus=${GPUS[*]} budget=$BUDGET"
echo "[q30] ${#CELLS[@]} cells: dims='$DIMS' arms='$ARMS'"

# Cells are claimed atomically rather than striped by index, so this script can be
# invoked once per GPU AS THAT GPU FREES -- which is how it actually gets used, because
# GPUs come free at staggered times. Index striping only works if every worker starts
# together and knows the full GPU list; two independent invocations would both begin at
# index 0 and run the same cell twice.
CLAIM_DIR="outputs/.q30_claims"
mkdir -p "$CLAIM_DIR"

for i in "${!GPUS[@]}"; do
  (
    g="${GPUS[$i]}"
    lock="$LOCK_DIR/${HOST}_gpu${g}"
    echo "q30-grid" > "$lock"
    trap 'rm -f "$lock"' EXIT
    for ((j = 0; j < ${#CELLS[@]}; j += 1)); do
      read -r name arm k mode <<< "${CELLS[$j]}"
      out="outputs/$name"
      if [ -f "$out/final.pt" ]; then
        echo "[q30] gpu$g: $name already complete -> skip"
        continue
      fi
      # atomic claim: only one worker can create the file. A crashed worker's claim is
      # removed by the trap below, so the cell returns to the pool on the next pass.
      if ! ( set -o noclobber; echo "$HOST gpu$g $$" > "$CLAIM_DIR/$name" ) 2>/dev/null; then
        echo "[q30] gpu$g: $name claimed by $(cat "$CLAIM_DIR/$name" 2>/dev/null) -> skip"
        continue
      fi
      trap 'rm -f "$lock"; rm -f "$CLAIM_DIR/'"$name"'"' EXIT
      echo "[q30] gpu$g: launching $name (arm=$arm d=$k coord=$mode)"
      mkdir -p "$out"
      # resubmission = resume: train_small.py auto-resumes from its own ckpt
      ./run_llm.sh "$g" "$name" \
          "${COMMON[@]}" \
          --grid_coord_dims "$k" --clrs_coord_mode "$mode" \
          --extra_json "${ARM_JSON[$arm]}" \
        || echo "[q30] gpu$g: $name FAILED (see $out/train.log)"
      # the startup guard must have fired, or the address never reached a layer
      grep -q "COORD VERIFIED ACTIVE\|address correctly inert" "$out/train.log" \
        || echo "[q30] gpu$g: $name -- NO COORD VERIFICATION LINE, TREAT AS INVALID"
      # keep the claim only while the cell is unfinished; a finished cell is guarded by
      # final.pt from here on, and a failed one must go back into the pool
      [ -f "$out/final.pt" ] || rm -f "$CLAIM_DIR/$name"
      trap 'rm -f "$lock"' EXIT
    done
  ) &
done
wait
echo "[q30] all cells done on $HOST"
