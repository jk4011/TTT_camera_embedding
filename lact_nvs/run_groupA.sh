#!/usr/bin/env bash
# GROUP A seed completion: the seven standard-protocol runs that exist at s95 only.
#
#   ./run_groupA.sh <gpu>        # one worker; pulls cells until the pool is empty
#
# WHY IT MATTERS FOR THE PAPER, not just for tidiness: because GTA and PRoPE and the
# depth-4 pair sit at one seed, Table 1 currently has to be split into two blocks with
# two different baselines, and the caption has to say so. Finishing these collapses the
# blocks and removes the caveat. Each is 1.6 h at the standard protocol.
#
# Claims are atomic, like Q30's, because workers start at staggered times as the 80k
# runs ahead of them finish.
set -u
[ $# -ge 1 ] || { echo "usage: $0 <gpu>" >&2; exit 1; }
GPU=$1
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
CLAIM=outputs/.groupA_claims; mkdir -p "$CLAIM"
HOST="$(hostname -s)"

# name                    config                     seed
CELLS=(
  "q15_gta_in_s137        config/cam_gta_in.yaml     137"
  "q15_gta_in_s211        config/cam_gta_in.yaml     211"
  "q15_prope_orig_s211    config/cam_prope_orig.yaml 211"
  "fw4l_base_s137         config/cam_fw4l_base.yaml  137"
  "fw4l_base_s211         config/cam_fw4l_base.yaml  211"
  "fw4l_rot4_s137         config/cam_fw4l_rot4.yaml  137"
  "fw4l_rot4_s211         config/cam_fw4l_rot4.yaml  211"
)

for row in "${CELLS[@]}"; do
  read -r NAME CFG SEED <<< "$row"
  [ -f "outputs/$NAME/eval.json" ] && { echo "[gA] gpu$GPU: $NAME done -> skip"; continue; }
  if ! ( set -o noclobber; echo "$HOST gpu$GPU $$" > "$CLAIM/$NAME" ) 2>/dev/null; then
    echo "[gA] gpu$GPU: $NAME claimed by $(cat "$CLAIM/$NAME" 2>/dev/null) -> skip"; continue
  fi
  echo "[gA] gpu$GPU: launching $NAME (seed $SEED)"
  ./launch_exp.sh "$GPU" "$NAME" "$CFG" "$SEED"
  CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$NAME/model_0030000.pth" \
    --config "$CFG" > "outputs/$NAME/eval.log" 2>&1
  echo "[gA] gpu$GPU: $NAME eval exit=$?"
  [ -f "outputs/$NAME/eval.json" ] || rm -f "$CLAIM/$NAME"   # failed -> back in the pool
done
echo "[gA] gpu$GPU: pool drained"
