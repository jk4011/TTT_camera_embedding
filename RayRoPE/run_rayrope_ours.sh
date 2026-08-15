#!/bin/bash
# Q45: run RayRoPE's OWN codebase end-to-end on OUR gObjaverse renders.
#   ./run_rayrope_ours.sh <gpu> <arm>   arm in {rayrope, prope, ropeonrays}
# Purpose (user 2026-08-13): if their code reproduces their Table-1 ordering on our
# data, our earlier phase-collapse was port/testbed-side; if RoPE-on-rays collapses
# here too, the render distribution is confirmed as the cause with their own code.
set -u
GPU=$1; ARMKEY=$2
TAG=${TAG:-}
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
case "$ARMKEY" in
  rayrope)    POS_ENC="d_pj+0_3d";   EXTRA="--model_config.depth_type predict_dsig" ;;
  prope)      POS_ENC="prope";       EXTRA="" ;;
  ropeonrays) POS_ENC="global-0+inf"; EXTRA="" ;;
  *) echo "unknown arm $ARMKEY"; exit 1 ;;
esac
export OBJV_DIR=${OBJV_DIR:-/tmp/rayrope_objv}
export CO3D_DIR=/tmp/rayrope_objv CO3D_DEPTH_DIR=/tmp/rayrope_objv CO3D_ANNOTATION_DIR=/tmp/rayrope_objv
export RE10K_TRAIN_DIR=/tmp/rayrope_objv RE10K_TEST_DIR=/tmp/rayrope_objv RE10K_DIR=/tmp/rayrope_objv
export PYTHONPATH="$PWD/_stubs:$PWD"
export TRITON_CACHE_DIR=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/.cache_triton_rayrope
export TORCHINDUCTOR_CACHE_DIR=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/.cache_inductor_rayrope
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p results
STEPS=${STEPS:-80000}
CUDA_VISIBLE_DEVICES=$GPU RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29920 + GPU)) \
"$PY" nvs/trainval.py lvsm \
  --amp --amp_dtype ${AMP_DTYPE:-fp16} \
  --dataset objaverse \
  --objaverse_train_index_file "${TRAIN_INDEX:-assets/objaverse_index_train_context2.json}" \
  --dataset_batch_scenes 8 \
  --model_config.encoder.num_layers 6 \
  --model_config.encoder.layer.d_model 768 \
  --model_config.encoder.layer.nhead 16 \
  --model_config.encoder.layer.dim_feedforward 1024 \
  --model_config.encoder.layer.qk_norm \
  --model_config.pos_enc "$POS_ENC" $EXTRA \
  --max_steps $STEPS --test_every 20000 \
  --output_dir "results/ours-${ARMKEY}${TAG}" \
  > "results/ours-${ARMKEY}${TAG}.log" 2>&1
echo "[$ARMKEY] exited rc=$?"
