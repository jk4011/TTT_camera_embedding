#!/usr/bin/env bash
# Q37: PRoPE's OWN code on OUR gObjaverse renders -- one arm per invocation.
#   ./run_prope_repro.sh <gpu> <pos_enc: none|prope|gta>
#
# Their release recipe exactly (scripts/nvs.sh): 6 layers, d768, nhead 16, ffn 1024,
# qk_norm, fp16 AMP, bs 8 scenes, supervise 1, lr 4e-4 (default), 80k steps, 2 context
# views chosen by THEIR selector (frame distance 25-100 -> 25-39 on our 40-frame
# orbits). Only the data differs: our gobjaverse_wai renders via transforms.json
# conversion (poses verified opencv; blender2opencv pre-cancelled).
#
# Eval: their EvalDataset with our deterministic index (context [7,32] = frame
# distance 25, their training minimum -- the most favourable in-distribution
# geometry their regime produces; targets [15,20,27]), 500 held-out objects.
#
# WHAT IT DECIDES. F51: our sinusoidal TTT rotary is harmful on this data. If their
# attention-site group action GAINS here (prope/gta vs none), their Objaverse claim
# survives on measured wide geometry and the difference is encoding/site, NOT their
# renders. If it does NOT gain, the "favourably rendered Objaverse" suspicion gains
# real weight -- their published gain would not reproduce on the one Objaverse
# render set whose geometry is public.
set -u
GPU=$1; ARM=$2
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
OUT="results/gobj-${ARM}"
mkdir -p results
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29890 + GPU)) \
"$PY" nvs/trainval.py lvsm \
  --amp --amp_dtype fp16 \
  --dataset_batch_scenes 8 \
  --dataset_supervise_views 1 \
  --model_config.encoder.num_layers 6 \
  --model_config.encoder.layer.d_model 768 \
  --model_config.encoder.layer.nhead 16 \
  --model_config.encoder.layer.dim_feedforward 1024 \
  --model_config.encoder.layer.qk_norm \
  --max_steps 80000 --test_every 20000 \
  --test_index_fp evaluation_index_gobj.json \
  --model_config.ray_encoding plucker \
  --model_config.pos_enc "$ARM" \
  --output_dir "$OUT" > "results/gobj-${ARM}.log" 2>&1
echo "[$ARM] exited rc=$?"
