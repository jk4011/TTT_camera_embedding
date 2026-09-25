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
    # Objaverse: the ray:point = 1:3 split of the rotary budget (dp_lin_pdir31_vo_both.yaml, user 2026-09-25)
    ("CaPET (ours)", "#d62728", ("re10k_dplin_pdir_vo_s137", "gobj_dplin31_pdir_vo_s137",
                                 "dl3dvu_dplin_pdir_vo_s137")),
]
PANELS = [("RealEstate10K", 0), ("DL3DV", 2), ("Objaverse", 1)]   # paper order (user, 2026-09-25)

p = argparse.ArgumentParser()
p.add_argument("--views", nargs="+", type=int, default=[4, 8, 16, 32])
p.add_argument("--out", type=str, default="../paper_overleaf/figs/input_scale.pdf")
p.add_argument("--trained_at", type=int, default=8)
args = p.parse_args()


def psnr(exp, v):
    # Objaverse: FIXED targets 7, 12, 32, 37 (eval_fixed_targets.py, F103). The standard selection moves the
    # targets with the input count and always keeps one target that duplicates an input view.
    if exp.startswith("gobj"):
        f = f"outputs/{exp}/eval_fixedT_nv.json"
        try:
            return json.load(open(f))["psnr"][str(v)]
        except Exception:
            return None
    f = f"outputs/{exp}/eval_paper_nv{v}.json"
    if not os.path.exists(f):
        return None
    try:
        return json.load(open(f))["psnr"]
    except Exception:
        return None


# Roman type like the paper body (Liberation Serif = Times metrics), a bit larger than before, and the legend as a
# vertical list in its own column on the left (user 2026-09-25), so it never covers the PSNR label.
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Liberation Serif", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],   # Times metrics, TrueType
                     "mathtext.fontset": "stix", "font.size": 12.5,
                     "axes.titlesize": 13.5, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "pdf.fonttype": 42, "ps.fonttype": 42})   # embed TrueType, not Type 3 (venue font checks)
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.3))
fig.subplots_adjust(left=0.215, right=0.995, bottom=0.17, top=0.9, wspace=0.26)
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
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("PSNR")
h, l = axes[0].get_legend_handles_labels()
if l:
    fig.legend(h, l, loc="center left", frameon=False, fontsize=12.5, labelspacing=1.3,
               handlelength=2.0, borderaxespad=0.0, bbox_to_anchor=(0.0, 0.53))
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
fig.savefig(args.out, bbox_inches="tight")
print("saved ->", args.out)
for label, _, exps in ARMS:
    row = " ".join(f"v{v}={psnr(exps[d], v):.2f}" if psnr(exps[d], v) else f"v{v}=--"
                   for d in range(3) for v in args.views)
    print(f"{label:14s} {row}")
