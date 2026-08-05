#!/usr/bin/env bash
# Q31 repair: q31_attnnope_nope_s43 blew up mid-training and its number is unusable.
#
#   ./run_q31_nope_rerun.sh <gpu>
#
# WHAT HAPPENED. The cell tracks its siblings exactly for the first half and then
# turns around and climbs:
#
#   step        2k      14k     26k     38k     50k     62k     74k     86k
#   nope_s43  116.76   32.51   27.25   24.67   22.78   26.20   28.50   30.40   <-- diverges
#   nope_s42  118.84   32.80   27.49   24.90   22.94   21.37   20.26   19.72
#   in_s43    116.35   32.34   27.10   24.58   22.62   21.06   19.95   19.42
#
# So it is a mid-training instability around 50-62k, not a bad initialisation and not
# a data-order effect: through 50k it is indistinguishable from the arms that finished
# fine. Averaging 30.40 in would manufacture a large fake "NoPE is much worse under
# attn_nope" effect out of one blown-up run.
#
# WHAT THIS RERUN IS, STATED HONESTLY. data_seed stays 43 so the cell remains paired
# with in/h/both at that data seed; only the init seed changes (42 -> 44). A different
# init MAY not avoid the instability, and that is the point: a model with no explicit
# positional code anywhere (attn_nope + ttt_nope) being less stable is a REAL property,
# not noise. If this rerun also diverges, we report the instability itself rather than
# a perplexity.
#
# The diverged run is kept at outputs/q31_attnnope_nope_s43 -- it is evidence, not a
# mistake to be deleted. This one lands in q31_attnnope_nope_s43b.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <gpu>" >&2; exit 1; }
GPU=$1
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME=q31_attnnope_nope_s43b
[ -f "outputs/$NAME/final.pt" ] && { echo "[q31r] $NAME already complete"; exit 0; }
mkdir -p "outputs/$NAME"
# COMMON copied verbatim from run_attnnope_grid.sh; only the seeds differ.
./run_llm.sh "$GPU" "$NAME" \
  --seq_len 4096 --window_size 1024 --lact_chunk_size 1024 \
  --hidden_size 768 --num_hidden_layers 12 --num_attn_heads 12 --num_lact_heads 4 \
  --bs 8 --grad_accum 1 --lr 3e-4 \
  --token_budget 3000000000 \
  --data_seed 43 --seed 44 \
  --val_every 2000 --log_every 100 --val_bs 8 --val_tokens 2000000 \
  --extra_json '{"attn_nope": true, "ttt_nope": true}'
echo "[q31r] $NAME exit=$?"
