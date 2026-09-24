#!/bin/bash
# node2: Table 4 with the linear absolute-depth head (dpt_lin + dpt_abs), 9 cells ONE AT A TIME, row by row
# (input+hidden -> input only -> hidden only), so each table row completes as a unit. Rerun after a reset.
main() {
  local c=() v d
  for v in both in h; do for d in re10k gobj dl3dvu; do
    c+=("$d:${d}_dplin_pdir_${v}_s137:config/dp_lin_pdir_${v}.yaml")
  done; done
  GPU=${GPU:-0} NODE=${NODE:-node2} SEQ_LOG=outputs/queue_table4_lin.log \
    bash /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs/run_seq.sh "${c[@]}"
}
main "$@"
