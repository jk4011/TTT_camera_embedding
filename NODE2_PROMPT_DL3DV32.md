# NODE2 prompt — Q34: DL3DV at 32 input views, four arms SEQUENTIALLY on your free GPU

`git pull` first. User assignment 2026-08-06: one GPU is free on node2; run the four
Q34 arms on it sequentially (~5-6 h each, ~22-24 h total plus evals).

## What this answers

F50 showed `both` fails to compose on DL3DV at 8 views (both - best single = -0.148,
t = -9.74) on the SAME LVSM backbone where RE10K composes. The user's hypothesis: at
32 input views the window is packed with overlapping poses, so camera addressing
matters more and the failure might reverse.

Measured through the eval loader, 8 -> 32 views leaves the median pairwise angle at
~39 deg (the window sets it) but drops the NEAREST-NEIGHBOUR angle 12.8 -> 4.7 deg,
i.e. locally RE10K. So this grid cleanly separates two accounts:
- retrieval leans on nearest-neighbour pairs -> `both` recovers at 32 views;
- the typical pair governs -> the failure stays. (F48, tttLRM eval-only, points this
  way, but its phases were trained at 8 views; only training at 32 answers it.)

## Steps

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope && git pull
cd lact_nvs

# 1. reshard DL3DV into YOUR node's /tmp (node-local; ~40-60 min).
#    Use fewer workers if video training is loading your CPUs.
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
D=/NHNHOME/WORKSPACE/26msit001_A/V-LAB/Datasets/dl3dv/dl3dv_undistorted_960
$PY data_preprocess/reshard_dl3dv.py --src $D/train --odir /tmp/dl3dv/train \
    --index /tmp/dl3dv/train_index.json --workers 32
$PY data_preprocess/reshard_dl3dv.py --src $D/test  --odir /tmp/dl3dv/test \
    --index /tmp/dl3dv/test_index.json  --workers 16

# 2. claim your free GPU (check nvidia-smi AND the lock dir), then:
echo "dl3dv32 seq grid" > ../lact_nvs/outputs/.gpu_locks/$(hostname -s)_gpu<N>
./run_dl3dv_v32_grid.sh "<N>"        # ONE gpu -> the script runs the 4 arms sequentially
```

Launch as your own background Bash task (run_in_background) so you get completion
notifications; `setsid` if you might restart your session. The script resumes per arm
(skips arms with eval.json; train.py resumes from its outputs dir), so a mid-run
interruption costs at most the current arm's progress since its last checkpoint.

## Protocol notes (shared by all four arms; state them when reporting)

- 32 input + 8 target (num_all_views 40), bs_per_gpu 8 (16 does not fit at 40 views).
- Eval: the same 140-scene DL3DV test split as F50, but 32 uniform inputs. These
  numbers are NOT comparable to F50's 8-view numbers -- different input scale --
  only to each other.
- Everything else the F50 recipe: 30k, lr 1e-4, LPIPS from 5k, seed 95.

## Report as F53

Paired per-scene vs dl3dv32_base_s95, n=140, and THE number:
**(both - max(input, hidden))**, next to F50's -0.148. Positive = view density
restores composition (the user's hypothesis); still negative = the typical-pair
geometry governs. Append to RESULTS_DOSSIER.md, push. Paper is under FREEZE.
