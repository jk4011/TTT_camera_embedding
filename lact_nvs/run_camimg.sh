#!/usr/bin/env bash
# PRoPE-style split rotary: half the budget on camera, half on in-view 2D position.
#
#   ./run_camimg.sh <gpu> [seed]
#
# WHY. F34 found that PRoPE's entire gain in this stack came from its IMAGE-coordinate
# ropes (+0.379) while its projective transform cost -0.294. Our own ladder spends 100%
# of its budget on the 6 Plucker coordinates and nothing on where inside a view a token
# sits. This asks whether that allocation is right.
#
# BUDGET-MATCHED, verified not asserted: 6*F_cam + 2*F_img = 6*10 + 2*33 = 126 pairs,
# exactly the 6*21 = 126 of qk_rope_cam. 252 of 256 head dims rotated in both, 98.4%.
# So the comparison is pure ALLOCATION -- same number of rotated dimensions, same
# frequency range, same everything else.
#
# Comparator: pra_hi (input F21, no hidden site), s95 = 22.333, 3-seed 22.348 +- 0.033.
# Report paired per-scene against THAT, not against base.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <gpu> [seed]" >&2; exit 1; }
GPU=$1; SEED=${2:-95}
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
EXP=camimg_s${SEED}; CFG=config/cam_camimg.yaml
[ -f "outputs/$EXP/eval.json" ] && { echo "[camimg] $EXP already done"; exit 0; }
./launch_exp.sh "$GPU" "$EXP" "$CFG" "$SEED"
CUDA_VISIBLE_DEVICES=$GPU $PY eval.py --load "outputs/$EXP/model_0030000.pth" \
  --config "$CFG" > "outputs/$EXP/eval.log" 2>&1
echo "[camimg] eval exit=$?"
