#!/bin/bash
# tab:ccv camera accuracy for one generated set: frames -> COLMAP -> RotErr/TransErr/CamMC.
# CPU only (pycolmap in its own venv), so it runs alongside the GPU jobs. `--which gt` scores
# the REAL target-camera video through the same pipeline: that row is the metric's floor
# (~2 deg), and every generated number is read against it. Usage: run_ccv_campose.sh <exp>
main() {
  local EXP=$1
  local REPO=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope
  local LOG=$REPO/lact_nvs/outputs/queue_ccv.log
  local GEN=$REPO/lact_ar_video/outputs/eval/gen_${EXP}_13999
  local PREP=$REPO/lact_ar_video/outputs/eval/campose_${EXP}_13999
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
  export PYTHONPATH="$REPO/lact_ar_video${PYTHONPATH:+:$PYTHONPATH}"
  cd $REPO/lact_ar_video/minVid
  if [ ! -f "$PREP/pairs.json" ]; then
    echo "$(date '+%F %T') campose $EXP: extracting frames and conditioning poses" >> $LOG
    /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/.venv_llm/bin/python eval_ccv_campose_prep.py \
      --gen_dir "$GEN" --out "$PREP" >> "$PREP.log" 2>&1
  fi
  # the GT floor is the same for every method (same pairs, same real videos): run it once
  for WHICH in ${WHICHS:-gen gt}; do
    echo "$(date '+%F %T') campose $EXP: colmap on $WHICH" >> $LOG
    /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/sfm/bin/python eval_ccv_campose.py \
      --prep "$PREP" --which $WHICH --matcher exhaustive >> "$PREP.log" 2>&1
    echo "$(date '+%F %T') campose $EXP $WHICH done (rc=$?)" >> $LOG
  done
}
main "$@"
