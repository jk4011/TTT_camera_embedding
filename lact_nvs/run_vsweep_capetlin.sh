#!/usr/bin/env bash
# Input-view sweep for fig:input-scale, CaPET arm only, with the FINAL recipe checkpoints
# (dpt_lin+dpt_abs, 2026-09-25). Copy of run_vsweep_paper.sh; the other four arms are already done.
# Original header:
# Input-view sweep for the paper figure fig:input-scale (2026-09-19, user item 2).
# Five arms -- NoPE / PRoPE / GTA / RayRoPE (sigma0=3, world-frame) / CaPET (final recipe) --
# re-evaluated at several input-view counts with NO additional training. Seed 137 throughout.
#
#   ./run_vsweep_paper.sh <dataset: re10k|gobj|dl3dvu> <gpu> [views]   (default 4 8 16 32, user 2026-09-19)
#
# A fixed --min_frames keeps the scene set identical across view counts, so per-scene paired
# deltas stay valid. Writes outputs/<exp>/eval_paper_nv<V>.json and skips finished ones.
DATA_ROOT=${DATA_ROOT:-/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/reshard}
set -u
DS=$1; GPU=$2
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
REPO_ROOT="$(cd .. && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache_triton_nvs" TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache_inductor_nvs" TORCHINDUCTOR_COMPILE_THREADS=1

declare -A CFG=(
  [nope]=config/lact_l6_d256_p16.yaml
  [prope]=config/cam_prope_orig.yaml
  [gta]=config/cam_gta_in.yaml
  [rayrope]=config/rayrope_ttt_world.yaml
  [capet]=config/dp_lin_pdir_vo_both.yaml
)
WIDE=0
case "$DS" in
  re10k)
    declare -A EXP=( [nope]=base_s137_re [prope]=re10k_prope_s137 [gta]=re10k_gta_s137 \
                     [rayrope]=re10k_rayropew_s137 [capet]=re10k_dplin_pdir_vo_s137 )
    DP=$DATA_ROOT/re10k/test_index.json; NSC=256; EXTRA=(--min_frames 52); VIEWS=${3:-"4 8 16 32"} ;;
  gobj)
    declare -A EXP=( [nope]=gobj_base_s137_re [prope]=gobj_prope_s137 [gta]=gobj_gta_s137 \
                     [rayrope]=gobj_rayropew_s137 [capet]=gobj_dplin_pdir_vo_s137 )
    DP=$DATA_ROOT/gobj/test_index.json; NSC=500; EXTRA=(--min_frames 36); VIEWS=${3:-"4 8 16 32"} ;;   # 40-frame orbits
  dl3dvu)
    declare -A EXP=( [nope]=dl3dvu_base_s137_re [prope]=dl3dvu_prope_s137 [gta]=dl3dvu_gta_s137 \
                     [rayrope]=dl3dvu_rayropew_s137 [capet]=dl3dvu_dplin_pdir_vo_s137 )
    DP=$DATA_ROOT/dl3dv/test_index.json; NSC=140; EXTRA=(--min_frames 52 --image_size 256 448); VIEWS=${3:-"4 8 16 32"}; WIDE=1 ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
[ -f "$DP" ] || { echo "FATAL: $DP missing"; exit 1; }

for V in $VIEWS; do
  if [ "$V" -ge 32 ]; then BS=2; elif [ "$V" -ge 16 ]; then BS=4; else BS=8; fi
  [ "$WIDE" = "1" ] && [ "$V" -ge 16 ] && BS=1
  for ARM in capet; do
    E=${EXP[$ARM]}; CK=outputs/$E/model_0030000.pth; OUT=outputs/$E/eval_paper_nv${V}.json
    [ -f "$OUT" ] && { echo "[$DS gpu$GPU] $ARM v$V done"; continue; }
    [ -f "$CK" ] || { echo "[$DS gpu$GPU] $ARM v$V SKIP: no checkpoint $CK"; continue; }
    echo "[$DS gpu$GPU] $ARM v$V start $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "$CK" --config "${CFG[$ARM]}" \
      --data_path "$DP" --num_scenes $NSC "${EXTRA[@]}" --num_input_views "$V" --num_target_views 4 --bs $BS \
      --out "$OUT" > "outputs/$E/eval_paper_nv${V}.log" 2>&1
    echo "[$DS gpu$GPU] $ARM v$V exit=$? $(grep -h 'PSNR:' outputs/$E/eval_paper_nv${V}.log | tail -1)"
  done
done
echo "[$DS gpu$GPU] sweep done $(date +%H:%M:%S)"
