"""Compose fig:comparison_nvs from rows rendered by render_compare.py (2026-09-19).

Each row is one (dataset directory, scene) pair, so a single figure can mix datasets.
Reads the per-scene PNGs render_compare.py already wrote and lays them out as
rows x (methods + GT), with the dataset name on the left and per-image PSNR inset.

  python compose_figure.py --rows "outputs/_fig/re10k:228:RealEstate10K" \
      "outputs/_fig/gobj:88:Objaverse" ... --out ../paper_overleaf/figs/comparison_nvs.pdf
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

COLS = ["No Encoding", "GTA", "PRoPE", "RayRoPE", "CaPET (ours)", "GT"]
# eval.json of each row's methods, for the PSNR inset (per scene, mean over target views)
EVAL = {
    "re10k": ["base_s137_re", "re10k_gta_s137", "re10k_prope_s137", "re10k_rayropew_s137",
              "re10k_dplin_pdir_vo_s137"],
    "gobj": ["gobj_base_s137_re", "gobj_gta_s137", "gobj_prope_s137", "gobj_rayropew_s137",
             "gobj_dplin_pdir_vo_s137"],
    "dl3dvu": ["dl3dvu_base_s137_re", "dl3dvu_gta_s137", "dl3dvu_prope_s137",
               "dl3dvu_rayropew_s137", "dl3dvu_dplin_pdir_vo_s137"],
}

p = argparse.ArgumentParser()
p.add_argument("--rows", nargs="+", required=True, help='each "dir:scene:RowLabel"')
p.add_argument("--out", type=str, default="../paper_overleaf/figs/comparison_nvs.pdf")
p.add_argument("--dpi", type=int, default=220)
args = p.parse_args()

rows = []
for spec in args.rows:
    d, scene, label = spec.split(":")
    scene = int(scene)
    hit = glob.glob(os.path.join(d, f"scene{scene:04d}_t*_GT.png"))
    assert hit, f"no rendered images for scene {scene} in {d}"
    t = int(re.search(r"_t(\d+)_GT", hit[0]).group(1))
    key = os.path.basename(d.rstrip("/"))
    psnr = {}
    for col, exp in zip(COLS[:-1], EVAL.get(key, [])):
        f = f"outputs/{exp}/eval.json"
        if os.path.exists(f):
            v = json.load(open(f)).get("per_scene_psnr")
            if v and scene < len(v):
                psnr[col] = v[scene]
    rows.append((d, scene, t, label, psnr))

h0, w0 = np.array(Image.open(os.path.join(rows[0][0], f"scene{rows[0][1]:04d}_t{rows[0][2]}_GT.png"))).shape[:2]
fig, axes = plt.subplots(len(rows), len(COLS),
                         figsize=(len(COLS) * 1.9, len(rows) * 1.9 * h0 / w0))
axes = np.atleast_2d(axes)
for ri, (d, scene, t, label, psnr) in enumerate(rows):
    for ci, col in enumerate(COLS):
        ax = axes[ri, ci]
        ax.imshow(Image.open(os.path.join(d, f"scene{scene:04d}_t{t}_{col}.png")))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if ri == 0:
            ax.set_title(col, fontsize=8, pad=3)
        if ci == 0:
            ax.set_ylabel(label, fontsize=7)
        if col in psnr:
            ax.text(0.03, 0.04, f"{psnr[col]:.2f}", transform=ax.transAxes, fontsize=6,
                    color="white", va="bottom", ha="left",
                    bbox=dict(facecolor="black", alpha=0.55, pad=0.8, edgecolor="none"))
plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.03, right=0.997, top=0.965, bottom=0.005)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
png = os.path.splitext(args.out)[0] + ".png"
fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
print("saved ->", args.out, "and", png)
