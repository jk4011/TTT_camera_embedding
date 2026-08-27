#!/bin/bash
# Single-chunk video ablation: 4 arms, one per GPU, 20k steps.
# base/input use the plain kernel single_chunk path; hidden/both use the hidden kernel.
set -u
i=0
for arm in base ttt_in ttt_h ttt; do
  setsid nohup ./run_video.sh $i configs/ar/video2_${arm}_sc.yaml \
    -s max_fwdbwd_passes 20000 -s exp_name video2_${arm}_sc \
    > outputs/video2_${arm}_sc.launch.log 2>&1 < /dev/null &
  i=$((i+1))
done
echo "4 single-chunk arms launched on gpu 0-3"
