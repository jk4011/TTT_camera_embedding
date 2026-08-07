# NODE3 prompt — Q43: video re-run matched to the current TTT-RoPE recipe

`git pull` first. Node3 is free; this is 3 cells, ONE GPU EACH, ~1 day at the v20k
protocol. See VIDEO_RERUN_QUEUE.md for the full rationale and the PRE-REGISTERED
PREDICTION (read it BEFORE any number: under the content-tax account the full-coverage
arm may go NEGATIVE, and F21/F22's null may have been the 50% frac shielding half the
dims).

## What changed in code (already committed)

F21/F22 tested hidden-only + theta ladder + frac 0.5. ar_lact_swa_repeat.py now
supports `ttt_input_rope` with the (t,y,x) GRID carrier (cam_phase_mode "none"):
same theta-ladder family sized to the fast head dim, phases built with the identical
rope_apply_ar carrier trick, so the token->(t,y,x) mapping matches the attention rope
exactly. Three configs are ready:

| cell | config | arms |
|---|---|---|
| video2_base | configs/ar/video2_base.yaml | no TTT rotary (paired reference) |
| video2_ttt  | configs/ar/video2_ttt.yaml  | input+hidden, frac 0.98 |
| video2_ttt_frac | configs/ar/video2_ttt_frac.yaml | input+hidden, frac 0.5 |

## Setup (node-local staging -- /tmp dies with the node)

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope && git pull
mkdir -p /tmp/wan_ckpt
cp /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/wan_ckpt/* /tmp/wan_ckpt/
# dataset is read from lustre directly (data_root in the configs) -- no staging needed
cd lact_ar_video
./run_video.sh 0 configs/ar/video2_base.yaml &
./run_video.sh 1 configs/ar/video2_ttt.yaml &
./run_video.sh 2 configs/ar/video2_ttt_frac.yaml &
```

Launch as YOUR background Bash tasks (run_in_background) so you get completion
notifications; setsid if you might restart. Verify within 10 min:
1. all three train logs advance and losses are finite;
2. the ttt arms actually enter the grid-carrier input-rope path (base must NOT);
3. deterministic-noise seeds match across arms (the v20k protocol pairs per-step
   losses -- if the noise streams differ the pairing is void).

## Smoke first (10 min, before the full runs)

./run_video.sh 0 configs/ar/video2_ttt.yaml -s max_fwdbwd_passes 30
must reach step 30 with finite loss. If it crashes in the grid-carrier input-rope
path, report the traceback and STOP -- do not burn a day on a broken arm.

## Report

v20k protocol = F22: 20k steps, paired per-step loss over the last 2.5k (and the
second half), deterministic noise. Report BOTH arms vs base with t-stats, as F62.
Either sign is informative (see the pre-registration). Paper is under FREEZE.
