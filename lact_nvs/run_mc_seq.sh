#!/usr/bin/env bash
# Multi-chunk NVS grid: four arms, SEQUENTIAL on one GPU, then evaluate each finished
# model at n = 1, 2, 4, 8 (NODE2_PROMPT_MC_RESTART.md).
# Usage: ./run_mc_seq.sh <gpu>
#
# All four arms must come from ONE commit — the rotary fusion (6d2f388) changes bf16
# rounding by ~1 ULP, so a run from before it is not comparable with one from after at
# the precision the paired comparison assumes. That is why this drives all four in a
# single script instead of launching them one at a time by hand.
#
# Each arm's eval runs right after its own training, on the same GPU, so a Slurm stop
# costs at most one arm rather than the whole grid.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="${1:-3}"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs"
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" mc_eval

ARMS="mc_base mc_in mc_h mc_both"

for arm in $ARMS; do
    cfg="config/${arm}.yaml"
    ckpt="outputs/${arm}/model_0030000.pth"

    if [ -f "$ckpt" ]; then
        echo "[mc] $arm already trained, skipping"
    else
        echo "[mc] === training $arm on gpu$GPU  $(date '+%F %T') ==="
        ./launch_mc.sh "$GPU" "$arm" "$cfg" 95
        if [ ! -f "$ckpt" ]; then
            echo "[mc] $arm FAILED to produce $ckpt — stopping the sequence"
            tail -5 "outputs/${arm}/train.log" 2>/dev/null
            exit 1
        fi
    fi

    for n in 1 2 4 8; do
        out="mc_eval/${arm}_n${n}.json"
        [ -f "$out" ] && { echo "[mc] $arm n=$n already evaluated"; continue; }
        echo "[mc] eval $arm n=$n"
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" eval.py --load "$ckpt" --config "$cfg" \
            --num_scenes 256 --num_input_views 32 --num_target_views 4 \
            --ttt_num_chunks "$n" --out "$out" > "mc_eval/${arm}_n${n}.log" 2>&1 \
          || echo "[mc] $arm n=$n EVAL FAILED (see mc_eval/${arm}_n${n}.log)"
    done
    echo "[mc] $arm done  $(date '+%F %T')"
done

echo "===== MC SEQUENCE DONE $(date '+%F %T') ====="
for arm in $ARMS; do
    for n in 1 2 4 8; do
        f="mc_eval/${arm}_n${n}.json"
        [ -f "$f" ] && "$PY" -c "import json;d=json.load(open('$f'));print(f'$arm n=$n PSNR {d[\"psnr\"]:.4f} LPIPS {d[\"lpips\"]:.4f}')"
    done
done
