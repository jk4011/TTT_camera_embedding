#!/bin/bash
# When Q46 launches (renders complete), reshard the new renders for OUR stack and
# start the TTT four-arm grid on GPUs 3-6 in parallel with Q46's GPUs 0-2.
T=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/objaverse/tools
L=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
until grep -q "Q46 LAUNCHED" "$T/chain_q46.log" 2>/dev/null; do sleep 300; done
/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python \
  "$L/data_preprocess/reshard_rayrope_renders.py" > "$L/outputs/reshard_vi.log" 2>&1
grep -q "RESHARD DONE" "$L/outputs/reshard_vi.log" || { echo "RESHARD FAILED"; exit 1; }
bash "$L/run_ttt_vi_grid.sh" "3 4 5 6" 95 > "$L/outputs/ttt_vi_grid.log" 2>&1
echo "TTT-VI CHAIN DONE $(date)"
