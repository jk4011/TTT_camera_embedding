# NODE3 prompt — Q37 replication: PRoPE's OWN code on OUR gObjaverse renders

`git pull` first. Node1 is already running all three arms once; your run is the
REPLICATION (their trainer has no seed control, so an independent run doubles as a
second seed). ~2 h/arm on one GPU each.

## Why

PRoPE reports +2.3 dB over the raymap baseline on Objaverse (their Table 1), but they
rendered Objaverse themselves and released neither renders nor the render script
(issue #12), the training view count is stated nowhere, and issue #13 suggests the
Objaverse runs used ~2x the compute of the RE10K ones. Our F51 finds every sinusoidal
rotary arm HARMFUL on the one Objaverse render set whose geometry is measured
(gobjaverse_wai, ~91 deg median). This run asks whether THEIR gain reproduces on that
measured geometry, using their own code, site (attention), and recipe.

## Steps

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope && git pull
# working copy (prope/ itself stays read-only; prope_run/ is gitignored)
cp -r prope prope_run && cd prope_run
sed -i 's|glob.glob("./data_processed/realestate10k/train/\*")|glob.glob("/tmp/gobj_prope/train/*")|' nvs/trainval.py
sed -i 's|folder = "./data_processed/realestate10k/test/"|folder = "/tmp/gobj_prope/test/"|' nvs/trainval.py
grep -c "/tmp/gobj_prope" nvs/trainval.py        # must print 2

# convert (writes tiny JSONs + image symlinks into YOUR node's /tmp; ~3 min)
/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python \
  ../lact_nvs/data_preprocess/convert_gobj_prope.py

# copy the launcher from node1's commit and run the three arms on three GPUs
cp ../lact_nvs/data_preprocess/run_prope_repro.sh . 2>/dev/null || true
# (if absent, the script is reproduced at the bottom of this file)
bash run_prope_repro.sh <gpuA> none &
bash run_prope_repro.sh <gpuB> prope &
bash run_prope_repro.sh <gpuC> gta &
```

Deps (tyro, torchmetrics, tensorboard) are already in the SHARED lvsm env.

## Report

Final-step PSNR/SSIM/LPIPS per arm from results/gobj-<arm>.log (their tester runs at
20k/40k/60k/80k). The number: prope - none and gta - none. Positive at roughly their
paper's margin -> their claim survives measured wide geometry (encoding/site is the
story). Flat or negative -> the unverifiable-renders concern is now backed by a
failed reproduction on public geometry. Append to RESULTS_DOSSIER.md as F54-repl,
push. Paper is under FREEZE.
