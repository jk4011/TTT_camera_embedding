#!/bin/bash
# rot_raw takes the next TWO GPUs to free: gpu5 after prope75, gpu6 after gobj_hga's
# successor finishes -- detected by nvidia-smi process count, not by run names.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCR=/tmp/claude-3943/-NHNHOME-WORKSPACE-26msit001-A-jinhyeok-TTT-rope/f3b76581-cf9a-4ec7-8166-d1b2ddfb9311/scratchpad/run_goal_cell.sh
launched_gobj=0; launched_re10k=0
while [ "$launched_gobj$launched_re10k" != "11" ]; do
  for g in 4 5 6 7; do
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader -i $g 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
      if [ "$launched_gobj" = 0 ]; then
        launched_gobj=1; echo "[$(date +%H:%M)] gpu$g -> gobj_rot_raw"
        setsid nohup bash "$SCR" $g gobj_rot_raw_s95 config/cam_rot_raw.yaml gobj > outputs/rot_raw_gobj.launch.log 2>&1 < /dev/null
      elif [ "$launched_re10k" = 0 ]; then
        launched_re10k=1; echo "[$(date +%H:%M)] gpu$g -> rot_raw (re10k)"
        setsid nohup bash "$SCR" $g rot_raw_s95 config/cam_rot_raw.yaml re10k > outputs/rot_raw_re10k.launch.log 2>&1 < /dev/null
      fi
      sleep 45
    fi
  done
  sleep 120
done
echo "both rot_raw cells launched"
