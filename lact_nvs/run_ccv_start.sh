#!/bin/bash
# Paper item 6 (CCV): start the camera-controlled video cells as soon as GPUs free, so nothing idles.
# One cell per GPU (the protocol the earlier ccv runs used). The baseline needs no new code, so it
# starts first; the CaPET cell is launched by run_ccv_capet.sh once its port is verified.
REPO=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope
LOCKS=$REPO/lact_nvs/outputs/.gpu_locks
LOG=$REPO/lact_nvs/outputs/queue_ccv.log
cd $REPO/lact_ar_video
export HF_HOME=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/.hf_cache          # /tmp is wiped on node reset
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
export C_INCLUDE_PATH=/usr/local/cuda/include
export PATH="/usr/local/cuda/bin:$REPO/.venv_llm/bin:$PATH"
export TRITON_CACHE_DIR="$REPO/.cache_triton" TORCHINDUCTOR_CACHE_DIR="$REPO/.cache_inductor"
export PYTHONPATH="$REPO/lact_ar_video${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"
GPU=$1; VARIANT=$2; EXP=$3
# the PE queue claims the card BEFORE launching us, so waiting on its lock would deadlock
[ "${PE_QUEUE_OWNED:-0}" = "1" ] || until [ ! -f $LOCKS/node1_gpu$GPU ]; do sleep 120; done
echo "ccv:$EXP" > $LOCKS/node1_gpu$GPU
echo "$(date '+%F %T') ccv $EXP starting on gpu$GPU (config abl_ccv_${VARIANT}.yaml)" >> $LOG
cd minVid
CUDA_VISIBLE_DEVICES=$GPU $REPO/.venv_llm/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=1 --master_port $((29610 + GPU)) \
  train.py "configs/ar/abl_ccv_${VARIANT}.yaml" -s exp_name "$EXP" \
  >> $REPO/lact_ar_video/outputs/${EXP}.log 2>&1
RC=$?
echo "$(date '+%F %T') ccv $EXP exited rc=$RC" >> $LOG
rm -f $LOCKS/node1_gpu$GPU
exit $RC
