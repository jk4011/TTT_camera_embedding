#!/bin/bash
# chain2.sh <exp> <gpu> <cfg> <next_exp> <next_cfg>
cd "$(dirname "$0")"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
exp=$1; gpu=$2; cfg=$3; nexp=$4; ncfg=$5
for i in $(seq 1 180); do
  [ -f outputs/$exp/model_0030000.pth ] && break; sleep 60
done
CUDA_VISIBLE_DEVICES=$gpu $PY eval.py --load outputs/$exp/model_0030000.pth --config $cfg > outputs/$exp/eval.log 2>&1
./launch_exp.sh $gpu $nexp $ncfg
