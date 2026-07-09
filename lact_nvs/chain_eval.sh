#!/bin/bash
# chain_eval.sh <exp> <gpu> <cfg> — wait for 30k ckpt, then eval only
cd "$(dirname "$0")"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
for i in $(seq 1 200); do
  [ -f outputs/$1/model_0030000.pth ] && break; sleep 60
done
CUDA_VISIBLE_DEVICES=$2 $PY eval.py --load outputs/$1/model_0030000.pth --config $3 > outputs/$1/eval.log 2>&1
