#!/bin/bash
cd "$(dirname "$0")"
PY=/NHNHOME/WORKSPACE/26msit001_T_B/POSTECH-CGLAB/anaconda3/envs/LVSM/bin/python
declare -A CFG=(
  [pra_h_hi]=config/cam_pra_h_hi.yaml
  [pra_h_vo]=config/cam_pra_h_vo.yaml
  [pra_sinc_h]=config/cam_pra_sinc_h.yaml
  [sinc_h]=config/cam_sinc_h.yaml
)
GPUS=(0 1 2 3)
i=0
for e in pra_h_hi pra_h_vo pra_sinc_h sinc_h; do
  (
    until [ -f outputs/$e/model_0030000.pth ]; do sleep 30; done
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY eval.py \
      --load outputs/$e/model_0030000.pth --config ${CFG[$e]} \
      > outputs/$e/eval.log 2>&1
  ) &
  i=$((i+1))
done
wait
