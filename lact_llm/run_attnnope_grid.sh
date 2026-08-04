#!/usr/bin/env bash
# Q31: does the TTT rotary earn more once the ATTENTION rotary is removed?
#
# Usage:  ./run_attnnope_grid.sh <gpu_list_csv>
#         ARMS="nope in" ./run_attnnope_grid.sh 0,1
#
# THE HYPOTHESIS. Our 1-D language null has always been explained as "language is
# content-addressed". There is a simpler candidate we never tested: LaCT's LM keeps
# a rotary on its sliding-window ATTENTION, so every local position already has an
# explicit relative code and the fast-weight rotary has little left to contribute.
#
# This is a cleaner lever than F36's window shrink:
#   window 1024 -> 128   changes attention CAPACITY  (absolute ppl degrades 18.40 -> 18.61)
#   attn_nope            removes the positional CHANNEL only, capacity untouched
#
# With attn_nope the TTT rotary becomes the model's ONLY explicit positional code.
# Prediction: the four arms, which sit at 18.58 +- 0.06 and are mutually
# indistinguishable at w128 (3 seeds), should SEPARATE here. If they do not, then
# "attention already supplies position" is refuted and the content-addressing
# explanation stands on its own.
#
# HONEST SCOPE, to be stated in any write-up:
#   * causal masking still leaks position, so this is "no explicit code", not
#     "position-free" (Kazemnejad et al.'s NoPE result depends on exactly that leak).
#   * absolute ppl will likely get WORSE -- we removed a useful code. Report this as a
#     channel-value decomposition, never as a SOTA claim.
#
# Only the attn-rope-OFF column runs here: the ON column is F27, same protocol
# (200M LaCT, 3B tokens fineweb-edu, data_seed 42, bs 8 x 4096, window 1024).
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <gpu_list_csv>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_TOP="$(dirname "$SCRIPT_DIR")"

IFS=',' read -r -a GPUS <<< "$1"
ARMS=${ARMS:-"nope in h both"}
BUDGET=${BUDGET:-3000000000}

LOCK_DIR="$REPO_TOP/lact_nvs/outputs/.gpu_locks"
HOST="$(hostname -s)"
mkdir -p "$LOCK_DIR"

# attn_nope is ON for every cell here; ttt_* selects the arm.
declare -A ARM_JSON=(
  [nope]='{"attn_nope": true, "ttt_nope": true}'
  [in]='{"attn_nope": true, "ttt_nope": false}'
  [h]='{"attn_nope": true, "ttt_nope": true, "ttt_hidden_rope": true, "ttt_hrope_gain": 1.0}'
  [both]='{"attn_nope": true, "ttt_nope": false, "ttt_hidden_rope": true, "ttt_hrope_gain": 1.0}'
)

CELLS=()
for arm in $ARMS; do CELLS+=("q31_attnnope_${arm} $arm"); done

# F27 protocol exactly, so the attn-rope-ON column is F27 and needs no re-run.
COMMON=(
  --seq_len 4096 --window_size 1024 --lact_chunk_size 1024
  --hidden_size 768 --num_hidden_layers 12 --num_attn_heads 12 --num_lact_heads 4
  --bs 8 --grad_accum 1 --lr 3e-4
  --token_budget "$BUDGET"
  --data_seed 42 --seed 42
  --val_every 2000 --log_every 100 --val_bs 8 --val_tokens 2000000
)

echo "[q31] host=$HOST gpus=${GPUS[*]} arms='$ARMS' budget=$BUDGET (attn rope OFF in every cell)"

for i in "${!GPUS[@]}"; do
  (
    g="${GPUS[$i]}"
    lock="$LOCK_DIR/${HOST}_gpu${g}"
    echo "q31-attnnope" > "$lock"
    trap 'rm -f "$lock"' EXIT
    for ((j = i; j < ${#CELLS[@]}; j += ${#GPUS[@]})); do
      read -r name arm <<< "${CELLS[$j]}"
      out="outputs/$name"
      if [ -f "$out/final.pt" ]; then
        echo "[q31] gpu$g: $name already complete -> skip"; continue
      fi
      echo "[q31] gpu$g: launching $name (arm=$arm)"
      mkdir -p "$out"
      ./run_llm.sh "$g" "$name" "${COMMON[@]}" --extra_json "${ARM_JSON[$arm]}" \
        || echo "[q31] gpu$g: $name FAILED (see $out/train.log)"
    done
  ) &
done
wait
echo "[q31] all cells done on $HOST"
