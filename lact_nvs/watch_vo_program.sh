#!/bin/bash
# The v/o composition program: as GPUs 4-7 free, launch in priority order.
#   1. gobj_vo_only   -- transport-alone number T (isolates what the ladder costs)
#   2. gobj_ttt_vo    -- TTT-RoPE + v/o: the user's hidden-increment question, wide
#   3. ttt_vo (re10k) -- same increment, narrow
# Detection by process count; claims via noclobber so a re-run cannot double-launch.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCR=/tmp/claude-3943/-NHNHOME-WORKSPACE-26msit001-A-jinhyeok-TTT-rope/f3b76581-cf9a-4ec7-8166-d1b2ddfb9311/scratchpad/run_goal_cell.sh
CLAIM=outputs/.vo_claims; mkdir -p "$CLAIM"
CELLS=("gobj_vo_only_s95 config/cam_vo_only.yaml gobj"
       "gobj_ttt_vo_s95  config/cam_ttt_vo.yaml  gobj"
       "ttt_vo_s95       config/cam_ttt_vo.yaml  re10k")
done_all() { for row in "${CELLS[@]}"; do set -- $row; [ -f "outputs/$1/eval.json" ] || return 1; done; }
until done_all; do
  for g in 4 5 6 7; do
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader -i $g 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] || continue
    for row in "${CELLS[@]}"; do
      set -- $row; EXP=$1; CFG=$2; DATA=$3
      [ -f "outputs/$EXP/eval.json" ] && continue
      ( set -o noclobber; echo "gpu$g $$" > "$CLAIM/$EXP" ) 2>/dev/null || continue
      echo "[$(date +%H:%M)] gpu$g -> $EXP"
      setsid nohup bash "$SCR" $g "$EXP" "$CFG" "$DATA" > "outputs/${EXP}.launch.log" 2>&1 < /dev/null
      sleep 45; break
    done
  done
  sleep 180
done
echo "[vo program] all three cells done"
