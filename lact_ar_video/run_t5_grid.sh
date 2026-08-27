#!/bin/bash
# T5-conversion 4-arm grid (arXiv:2605.02772 recipe): one arm per GPU, short
# finetuning per the paper's rapid-conversion premise (~2.6 h at 6.35 s/step).
set -u
STEPS=${STEPS:-1500}
i=0
for arm in base in h both; do
  setsid nohup ./run_video.sh $i configs/ar/video2_t5_${arm}.yaml \
    -s max_fwdbwd_passes $STEPS -s exp_name video2_t5_${arm} \
    > outputs/video2_t5_${arm}.launch.log 2>&1 < /dev/null &
  i=$((i+1))
done
echo "4 t5 arms launched on gpu 0-3 (${STEPS} steps)"
