#!/bin/bash
# table:diverse_fast_weight (paper item 4): the final recipe on deeper fast-weight inner models,
# RealEstate10K only, seed 137. Starts once the DL3DV ablation queue has launched everything.
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
export NODE=node1
LOG=outputs/queue_0919c.log
until grep -q "queue drained" outputs/queue_0919b.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') DL3DV queue drained; starting the fast-weight cells" >> $LOG
QUEUE=(
  "re10k_fw_mlp2_capet_s137 config/fw_mlp2_capet.yaml"
  "re10k_fw_fw3l_capet_s137 config/fw_fw3l_capet.yaml"
  "re10k_fw_fw4l_capet_s137 config/fw_fw4l_capet.yaml"
)
i=0
while [ $i -lt ${#QUEUE[@]} ]; do
  for g in 0 1 2 3; do
    [ $i -lt ${#QUEUE[@]} ] || break
    [ -f outputs/.gpu_locks/node1_gpu$g ] && continue
    set -- ${QUEUE[$i]}; exp=$1; cfg=$2
    if [ -f outputs/$exp/eval.json ]; then echo "$(date '+%F %T') skip $exp (done)" >> $LOG; i=$((i+1)); continue; fi
    EXTRA_ARGS=--actckpt setsid nohup ./run_re10k.sh $g $exp $cfg 137 > outputs/$exp.launch.log 2>&1 < /dev/null &
    echo "$(date '+%F %T') launched $exp on gpu$g" >> $LOG
    i=$((i+1)); sleep 45
  done
  sleep 60
done
echo "$(date '+%F %T') queue drained (fast-weight cells launched)" >> $LOG
