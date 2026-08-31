# NODE2 prompt — gObjaverse camera-embedding program, wave 1 (2026-08-31)

Paste this whole file into the node2 Claude session. `git pull` first.

## Context (2 min read)
The user has refocused the project on camera embedding for NVS and wants the method to work on
wide-baseline data (gObjaverse orbits, ~90 deg between views), where TTT-RoPE currently HURTS
(input −0.41 / hidden −0.57 / both −0.89 vs the NoPE baseline 22.193). The analysis and the full
hypothesis list are in `OBJ_ANALYSIS.md` (read §0, §4, §5). node1 runs wave-1 cells H1/H2/H7
(oracle 3D-point rotary, shell chord rotary, pose-free tokens + rot transport). **node2 runs the
four cells below**, all on the same protocol as F51 (`run_gobj.sh` fixes it).

## Setup (once per node reset)
```bash
bash /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/claude_portable/setup_node.sh   # if not done
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope && git pull
cd lact_nvs
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
SRC=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/gobjaverse_wai
# reshard gObjaverse into node-local /tmp (≈2 min total on 72 cores; test first, then train)
$PY data_preprocess/reshard_gobjaverse.py --src $SRC --odir /tmp/gobj/test  --index /tmp/gobj/test_index.json  --split test  --workers 16
$PY data_preprocess/reshard_gobjaverse.py --src $SRC --odir /tmp/gobj/train --index /tmp/gobj/train_index.json --split train --workers 56
ls /tmp/gobj/train | wc -l   # expect 19500 ; test 500
```
The GT-depth side files (oracle cells only) already live on lustre at
`/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/gobj_depth_patch/{train,test}`; nothing to build.

## The four cells (one per GPU, ~2 h each incl. eval; launch all four now)
```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
NODE=node2 setsid nohup ./run_gobj.sh 0 gobj_attn_nope_s95   config/gobj_attn_nope.yaml   95 > outputs/gobj_attn_nope_s95.launch.log   2>&1 < /dev/null &
NODE=node2 setsid nohup ./run_gobj.sh 1 gobj_attn_prope_s95  config/gobj_attn_prope.yaml  95 > outputs/gobj_attn_prope_s95.launch.log  2>&1 < /dev/null &
NODE=node2 setsid nohup ./run_gobj.sh 2 gobj_hrot_rotraw_s95 config/gobj_hrot_rotraw.yaml 95 > outputs/gobj_hrot_rotraw_s95.launch.log 2>&1 < /dev/null &
NODE=node2 setsid nohup ./run_gobj.sh 3 gobj_imgvo_himg_s95  config/gobj_imgvo_himg.yaml  95 > outputs/gobj_imgvo_himg_s95.launch.log  2>&1 < /dev/null &
```
What they are:
- `attn_nope` / `attn_prope`: DIAGNOSTIC CEILING. The TTT layer replaced by LaCT's own block-causal
  full attention (same 6L/d256, same world-frame Plücker tokens), without / with a faithful PRoPE
  on q/k/v/o. Tells us how much headroom camera conditioning has at this scale.
- `hrot_rotraw` (H4): rot_raw (+0.43, the best matrix cell) + orthogonal ROTATION action on the
  hidden address space ("one matrix action per address space" at wide baseline).
- `imgvo_himg` (H10): imgvo (+0.39, current best) + image-coordinate rotary at the HIDDEN site.

`run_gobj.sh` claims `outputs/.gpu_locks/node2_gpu<i>` itself and removes it on exit. It is
resumable: rerunning the same command skips a finished cell (eval.json) and train.py resumes from
the last checkpoint (saved every 10k) after an interruption. Launch as your own background Bash
tasks so you are notified; verify after 60 s that all four `outputs/<exp>/train.log` files are
advancing (`Iter 0000200` lines) and that nvidia-smi shows four processes.

## When each cell finishes
`outputs/<exp>/eval.json` appears. Paired numbers vs the baseline evaluated on the SAME test
index (node1 writes `outputs/gobj_base_s95/eval_v2.json`; if it is not there yet, wait for it —
do not compare against the old `eval.json`, its scene set may be off by one):
```bash
$PY paired_eval.py outputs/gobj_base_s95/eval_v2.json outputs/gobj_attn_nope_s95/eval.json outputs/gobj_attn_prope_s95/eval.json outputs/gobj_hrot_rotraw_s95/eval.json outputs/gobj_imgvo_himg_s95/eval.json --md
$PY paired_eval.py outputs/gobj_rot_raw_s95/eval_v2.json outputs/gobj_hrot_rotraw_s95/eval.json     # H4 vs its own base
$PY paired_eval.py outputs/gobj_imgvo_s95/eval_v2.json   outputs/gobj_imgvo_himg_s95/eval.json      # H10 vs its own base
```
Append the four rows (PSNR / LPIPS / SSIM, paired dPSNR, t, win%) to `NODE2_RESULTS.md` under a
heading "gObjaverse wave 1 (node2, 2026-08-31)" and push. Numbers only; node1 writes the dossier.
Paper is under FREEZE — do not touch `paper_overleaf/`.

## If GPUs free up before node1 sends wave 2
Run, in this order, one per free GPU (all already implemented and smoke-tested):
1. `NODE=node2 ./run_gobj.sh <g> gobj_camray_properaw_s95 config/gobj_camray_properaw.yaml 95`
   (H7 with translation-carrying projective transport)
2. `NODE=node2 ./run_gobj.sh <g> gobj_h_dpra42_s95 config/cam_h_dpra42.yaml 95`  (H5: hidden
   rotary on the update-induced path only; compare with gobj_hidden_s95 −0.57)
3. `NODE=node2 ./run_gobj.sh <g> gobj_shell_iso_in_s95 config/gobj_shell_iso_in.yaml 95`  (H2 iso)
