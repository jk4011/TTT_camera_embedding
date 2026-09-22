#!/bin/bash
# tab:ccv generation: waits for a ccv cell's step-13999 checkpoint AND for its trainer to have
# released the card, then generates the held-out pairs with the protocol every earlier ccv eval
# used (40 Euler steps, guide 1.0, per-pair seed 424242 + index, ccv_holdout_pairs_64.json).
# All 64 pairs are requested; the generator writes after every pair and skips pairs already
# done, so a resubmission after a node reset continues where it stopped. Methods are compared
# on the pairs they have in common. Usage: run_ccv_gen.sh <gpu> <variant> <exp>
main() {
  local GPU=$1 VARIANT=$2 EXP=$3
  local REPO=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope
  local LOCKS=$REPO/lact_nvs/outputs/.gpu_locks
  local LOG=$REPO/lact_nvs/outputs/queue_ccv.log
  local CKPT=$REPO/lact_ar_video/outputs/$EXP/seed_1/checkpoint_model_013999
  local OUT=$REPO/lact_ar_video/outputs/eval/gen_${EXP}_13999
  export HF_HOME=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/.hf_cache
  export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
  export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
  export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
  export C_INCLUDE_PATH=/usr/local/cuda/include
  export PATH="/usr/local/cuda/bin:$REPO/.venv_llm/bin:$PATH"
  export TRITON_CACHE_DIR="$REPO/.cache_triton" TORCHINDUCTOR_CACHE_DIR="$REPO/.cache_inductor"
  export PYTHONPATH="$REPO/lact_ar_video${PYTHONPATH:+:$PYTHONPATH}"
  until [ -d "$CKPT" ] && [ ! -f "$LOCKS/node1_gpu$GPU" ]; do sleep 120; done
  echo "gen:$EXP" > "$LOCKS/node1_gpu$GPU"
  echo "$(date '+%F %T') ccv gen $EXP@13999 starting on gpu$GPU" >> $LOG
  mkdir -p "$OUT"
  cd $REPO/lact_ar_video/minVid
  CUDA_VISIBLE_DEVICES=$GPU $REPO/.venv_llm/bin/python eval_ccv_generate.py \
    --config "configs/ar/abl_ccv_${VARIANT}.yaml" --ckpt "$CKPT" --out "$OUT" \
    --steps 40 --guide_scale 1.0 >> "$OUT/gen.log" 2>&1
  local RC=$?
  echo "$(date '+%F %T') ccv gen $EXP exited rc=$RC ($(ls "$OUT"/*_gen.mp4 2>/dev/null | wc -l) videos)" >> $LOG
  rm -f "$LOCKS/node1_gpu$GPU"
  exit $RC
}
main "$@"
