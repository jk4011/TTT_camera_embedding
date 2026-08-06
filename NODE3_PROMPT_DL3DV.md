# NODE3 prompt — Q32: LaCT-LVSM on DL3DV (the data-geometry test)

Paste this whole file as the first message to the Claude session on node3.
Everything you need is committed; start with `git pull`.

---

## The question, in one line

`both` composes on LVSM/RE10K (+0.97 over NoPE, best arm) but LOSES a third of the
single-site gain on tttLRM/DL3DV (F45) — and tttLRM differs from LVSM in backbone,
head, ladder AND data. This grid changes ONLY the data: same LaCT-LVSM, same configs,
same protocol, DL3DV instead of RE10K.

Measured motivation (2026-08-06): RE10K is forward-dolly video — median between-view
angle 7.2 deg. DL3DV is orbit scans — 61.1 deg. At those baselines ~50% of the DL3DV
ladder wraps between views (|dtheta| > pi) vs ~38% on RE10K; wrapped frequencies act
as decorrelation, and a second rotary site doubles that dose without adding
information.

**Pre-registered readings — write down before looking at any number:**
- `both` fails to compose here too → data geometry is the lever. "When do the two
  sites compose" becomes a claim about camera baseline. Paper-worthy either way.
- `both` composes fine on DL3DV+LVSM → geometry is NOT the cause; the suspects
  move back to tttLRM's side (24-layer depth, Gaussian+depth head).

## Steps

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope && git pull
cd lact_nvs

# 1. reshard DL3DV into the RE10K per-scene format on THIS node's /tmp
#    (~10,125 train scenes; expect ≤1 h with 64 workers; ~100 GB tmpfs)
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
D=/NHNHOME/WORKSPACE/26msit001_A/V-LAB/Datasets/dl3dv/dl3dv_undistorted_960
$PY data_preprocess/reshard_dl3dv.py --src $D/train --odir /tmp/dl3dv/train \
    --index /tmp/dl3dv/train_index.json --workers 64
$PY data_preprocess/reshard_dl3dv.py --src $D/test  --odir /tmp/dl3dv/test \
    --index /tmp/dl3dv/test_index.json  --workers 32

# 2. four arms, one per GPU (launch as your own background Bash task,
#    run_in_background, so you are notified; setsid if you might restart)
./run_dl3dv_grid.sh "0 1 2 3"
```

The grid script refuses to start without the indexes, resumes per-arm (skips arms
whose eval.json exists; train.py itself resumes from outputs/<exp>), and evaluates
on the SAME 140-scene DL3DV test split F45 used, 8 uniform inputs / 4 midpoint
targets over a 128-frame window.

## Protocol notes (differences you must state when reporting, all shared by the 4 arms)

- 10,125 train scenes vs RE10K's 66k → ~47 epochs over 30k steps vs ~7. Shared
  across arms, but absolute PSNR is NOT comparable to the RE10K table.
- Images are stored pre-sized to cover 256 (one extra resample vs RE10K's native
  640x360). Shared across arms.
- Eval-window geometry measured through the actual loader: median pairwise view
  angle 20.6 deg (vs RE10K ~7 deg at the same protocol) — the geometry contrast
  survives the 128-frame window, which is the point.
- These numbers are comparable ONLY within this grid, and to nothing else.

## Report

Paired per-scene deltas vs `dl3dv_base_s95` with t-statistics and win counts
(n=140), exactly like F45's table. The single number that answers the question:
**(both − max(input, hidden))** — negative means the second site subtracts, like
tttLRM; positive means it composes, like RE10K.

Append the result to `RESULTS_DOSSIER.md` as **F50**, delete nothing, push. The
paper is under FREEZE — do not touch paper claims.

## Verify before leaving it

1. Each arm's train.log shows a sane it/s within 10 min (IO comes from tmpfs; if
   it/s is far below ~4.5, something is reading lustre instead).
2. `grep "rotary"` in each cam arm's log — the cam_mode line must appear; base must
   NOT have one.
3. GPU locks: claim `outputs/.gpu_locks/<host>_gpu<i>` per the house convention.

---

# ADDENDUM — Q33: gobjaverse (object orbits), the third geometry point

The user has extended the assignment: after (or alongside, if you have 8 GPUs) the
DL3DV grid, run the same four arms on gobjaverse. Pull again first (`git pull`).

Geometry axis, all measured through the actual loader: RE10K 7.2 deg -> DL3DV
61.1 deg -> **gobjaverse 93.2 deg** median between-view angle. If `both`'s failure to
compose tracks this axis monotonically, the camera-baseline claim becomes a
dose-response curve.

Data: `/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/gobjaverse_wai` — 20,001
scenes, 40 frames each, 512x512 RGBA renders, WAI `scene_meta.json` (opencv, c2w
`transform_matrix`; +z verified to look at the origin). `reshard_dl3dv.py` already
understands this format (auto-detected per scene) and composites the alpha on white
per the GSO recipe. Verified end to end on this side: loader emits 93.2 deg median
angle, poses normalize to max|t| = 1, images land in [0,1].

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
G=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/gobjaverse_wai

# hold out the LAST 500 keys (sorted) as test; the rest train.
# symlink split first so the reshard sees two dirs:
mkdir -p /tmp/gobj_src/{train,test}
$PY - <<'EOP'
import os
G="/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/gobjaverse_wai"
ks=sorted(os.listdir(G)); ks=[k for k in ks if os.path.isdir(os.path.join(G,k))]
for k in ks[:-500]: os.symlink(os.path.join(G,k), f"/tmp/gobj_src/train/{k}")
for k in ks[-500:]: os.symlink(os.path.join(G,k), f"/tmp/gobj_src/test/{k}")
print(len(ks)-500, "train /", 500, "test")
EOP
$PY data_preprocess/reshard_dl3dv.py --src /tmp/gobj_src/train --odir /tmp/gobj/train \
    --index /tmp/gobj/train_index.json --workers 64
$PY data_preprocess/reshard_dl3dv.py --src /tmp/gobj_src/test  --odir /tmp/gobj/test \
    --index /tmp/gobj/test_index.json  --workers 32

./run_gobj_grid.sh "0 1 2 3"     # or "4 5 6 7" alongside the DL3DV grid
```

**Deviations from the standard protocol, forced by the data and shared by all arms
(state them in the report):**
- `--min_frames 40`: gobjaverse scenes have exactly 40 frames, under the default
  `num_views*3 = 45` filter. Without the flag the dataset filters to EMPTY. The
  launcher passes it; do not strip it.
- No official test split: last 500 sorted keys held out. Deterministic, disjoint.
- 40-frame scenes mean the train window is always the whole orbit.
- Absolute numbers comparable only within this grid.

Report as F51: paired per-scene vs `gobj_base_s95`, n=500, and the one number
**(both - max(input, hidden))**, alongside the same number from DL3DV (F50) and
RE10K. Three points, one axis.
