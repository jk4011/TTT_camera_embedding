#!/usr/bin/env bash
# Feasibility measurement for P0-W2: run train_small.py exactly as T5-T8 would
# (--data dna --bs 1 --window_size 128, default --val_bs 8) for a few steps at a
# given seq_len, while polling nvidia-smi for peak memory on that GPU.
# Usage: ./bench_seqlen.sh <gpu> <seq_len> [steps]
# Reports: peak reserved MiB, tok/s, OOM yes/no.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gpu="$1"; S="$2"; steps="${3:-20}"
exp="bench_s${S}"
rm -rf "outputs/$exp"

# poll peak memory for THIS gpu into a file (background)
peak_file="outputs/${exp}_peak.txt"
mkdir -p outputs
echo 0 > "$peak_file"
(
  while true; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -1)
    [ -n "$m" ] && [ "$m" -gt "$(cat "$peak_file")" ] 2>/dev/null && echo "$m" > "$peak_file"
    sleep 2
  done
) &
poller=$!

./run_llm.sh "$gpu" "$exp" --data dna --seq_len "$S" --bs 1 --window_size 128 \
    --steps "$steps" --log_every 5 --val_every "$steps" --save_every 0
code=$?
kill "$poller" 2>/dev/null

peak=$(cat "$peak_file")
oom="no"
grep -qiE "out of memory|CUDA out of memory" "outputs/$exp/train.log" 2>/dev/null && oom="YES"
tps=$(grep -oE "tokens/sec=[0-9,]+" "outputs/$exp/train.log" 2>/dev/null | tail -1 | tr -d 'a-z/=,' )
valline=$(grep -E "^VAL step=" "outputs/$exp/train.log" 2>/dev/null | tail -1)
echo "RESULT seq_len=$S exit=$code peak_MiB=$peak oom=$oom tok_per_s=${tps:-NA}"
echo "RESULT_VAL seq_len=$S ${valline:-NO-VAL-REACHED}"
