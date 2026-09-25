"""fig:input-scale -- PSNR vs number of input views, one panel per dataset (2026-09-19).

Reads the sweep results written by run_vsweep_paper.sh (outputs/<exp>/eval_paper_nv<V>.json)
and writes the paper figure. Arms and checkpoints must match run_vsweep_paper.sh.

  python plot_input_scale.py [--views 4 8 16 32] [--out ../paper_overleaf/figs/input_scale.pdf]

Missing cells are skipped, so the figure can be drawn while the sweep is still running.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [   # label, colour, per-dataset experiment names (re10k, gobj, dl3dvu)
    ("No Encoding", "#8c8c8c", ("base_s137_re", "gobj_base_s137_re", "dl3dvu_base_s137_re")),
    ("GTA",         "#6baed6", ("re10k_gta_s137", "gobj_gta_s137", "dl3dvu_gta_s137")),
    ("PRoPE",       "#3182bd", ("re10k_prope_s137", "gobj_prope_s137", "dl3dvu_prope_s137")),
    ("RayRoPE",     "#fd8d3c", ("re10k_rayropew_s137", "gobj_rayropew_s137", "dl3dvu_rayropew_s137")),
    # final recipe (linear depth layer, dpt_lin+dpt_abs), 2026-09-25; run_vsweep_capetlin.sh
    ("CaPET (ours)", "#d62728", ("re10k_dplin_pdir_vo_s137", "gobj_dplin_pdir_vo_s137",
                                 "dl3dvu_dplin_pdir_vo_s137")),
]
PANELS = [("RealEstate10K", 0), ("DL3DV", 2), ("Objaverse", 1)]   # paper order (user, 2026-09-25)

p = argparse.ArgumentParser()
p.add_argument("--views", nargs="+", type=int, default=[4, 8, 16, 32])
p.add_argument("--out", type=str, default="../paper_overleaf/figs/input_scale.pdf")
p.add_argument("--trained_at", type=int, default=8)
args = p.parse_args()


def psnr(exp, v):
    f = f"outputs/{exp}/eval_paper_nv{v}.json"
    if not os.path.exists(f):
        return None
    try:
        return json.load(open(f))["psnr"]
    except Exception:
        return None


fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1))
for ax, (title, di) in zip(axes, PANELS):
    for label, colour, exps in ARMS:
        xs, ys = [], []
        for v in args.views:
            y = psnr(exps[di], v)
            if y is not None:
                xs.append(v); ys.append(y)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=colour, label=label,
                zorder=3 if "ours" in label else 2)
    if not ax.lines:
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="0.6")
        ax.set_yticks([])
    ax.axvline(args.trained_at, color="0.8", lw=1, ls="--", zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(args.views)
    ax.set_xticklabels([str(v) for v in args.views])
    ax.set_xlabel("input views")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("PSNR")
h, l = axes[0].get_legend_handles_labels()
if l:
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
fig.savefig(args.out, bbox_inches="tight")
print("saved ->", args.out)
for label, _, exps in ARMS:
    row = " ".join(f"v{v}={psnr(exps[d], v):.2f}" if psnr(exps[d], v) else f"v{v}=--"
                   for d in range(3) for v in args.views)
    print(f"{label:14s} {row}")
