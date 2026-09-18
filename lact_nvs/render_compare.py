"""Qualitative comparison renderer (paper fig:comparison_nvs, 2026-09-19).

Renders the SAME held-out scenes and target views for several trained models and writes
(a) one PNG per (scene, method) and (b) a composed grid figure (rows = scenes,
columns = methods + GT) as .png and .pdf.

The dataset is built exactly as in eval.py (eval_mode, shuffle=False), so scene index i
here is scene i of every eval.json's per_scene_* arrays -- which is how --scenes is chosen
(see pick_scenes.py, which ranks scenes by the PSNR gap between two methods).

  python render_compare.py \
      --methods "NoPE=config/lact_l6_d256_p16.yaml,outputs/base_s137_re/model_0030000.pth" \
                "CaPET=config/dp_chan_pdir_vo_both.yaml,outputs/re10k_dpchan_pdir_vo_s137/model_0030000.pth" \
      --scenes 12,40,77 --out outputs/_fig/re10k
"""
import argparse
import os

import numpy as np
import omegaconf
import torch
from PIL import Image

from data_re10k import Re10KDataset
from model import LaCTLVSM

p = argparse.ArgumentParser()
p.add_argument("--methods", nargs="+", required=True,
               help='each "label=config.yaml,checkpoint.pth" (order = column order)')
p.add_argument("--scenes", type=str, required=True, help="comma-separated dataset indices")
p.add_argument("--data_path", type=str,
               default="/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/reshard/re10k/test_index.json")
p.add_argument("--num_scenes", type=int, default=256, help="must match the eval run (index alignment)")
p.add_argument("--num_input_views", type=int, default=8)
p.add_argument("--num_target_views", type=int, default=4)
p.add_argument("--image_size", nargs=2, type=int, default=[256, 256])
p.add_argument("--window", type=int, default=128)
p.add_argument("--min_frames", type=int, default=None)
p.add_argument("--pose_norm_mode", type=str, default="mean", choices=["mean", "norecenter"])
p.add_argument("--target", type=int, default=-1,
               help="which target view to show (-1 = the one with the largest spread across methods)")
p.add_argument("--out", type=str, required=True, help="output directory")
p.add_argument("--dpi", type=int, default=200)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
scene_ids = [int(s) for s in args.scenes.split(",") if s != ""]
methods = []
for m in args.methods:
    label, rest = m.split("=", 1)
    cfg, ckpt = rest.split(",", 1)
    methods.append((label, cfg, ckpt))

n_in, n_tg = args.num_input_views, args.num_target_views
dataset = Re10KDataset(
    args.data_path, num_views=n_in + n_tg, image_size=tuple(args.image_size),
    scene_pose_normalize=True, pose_norm_mode=args.pose_norm_mode, window=args.window,
    min_frames=args.min_frames, eval_mode=True, num_input_views=n_in,
    num_target_views=n_tg, max_scenes=args.num_scenes,
)
print(f"dataset: {len(dataset)} scenes; rendering {scene_ids}")
batch = {}
for k in dataset[0]:
    batch[k] = torch.stack([dataset[i][k] for i in scene_ids]).cuda()
input_dict = {k: v[:, :n_in] for k, v in batch.items()}
target_dict = {k: v[:, n_in:] for k, v in batch.items()}
gt = target_dict["image"].float().clamp(0, 1)                       # [S, n_tg, 3, H, W]

renders, psnrs = {}, {}
for label, cfg, ckpt in methods:
    model = LaCTLVSM(**omegaconf.OmegaConf.load(cfg)).cuda()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):
        r = model(input_dict, target_dict)
    r = r.float().clamp(0, 1)
    renders[label] = r.cpu()
    mse = ((r - gt) ** 2).flatten(2).mean(dim=2)                    # [S, n_tg]
    psnrs[label] = (-10.0 * torch.log10(mse)).cpu()
    print(f"{label:10s} per-scene PSNR (mean over targets): "
          + " ".join(f"{v:.2f}" for v in psnrs[label].mean(1).tolist()))
    del model
    torch.cuda.empty_cache()

# which target view to display per scene
tgt_idx = []
for s in range(len(scene_ids)):
    if args.target >= 0:
        tgt_idx.append(args.target)
    else:
        spread = torch.stack([psnrs[l][s] for l, _, _ in methods])   # [M, n_tg]
        tgt_idx.append(int((spread.max(0).values - spread.min(0).values).argmax()))

gt_cpu = gt.cpu()


def to_img(t):
    return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


cols = [l for l, _, _ in methods] + ["GT"]
for si, sid in enumerate(scene_ids):
    t = tgt_idx[si]
    for label, _, _ in methods:
        to_img(renders[label][si, t]).save(os.path.join(args.out, f"scene{sid:04d}_t{t}_{label}.png"))
    to_img(gt_cpu[si, t]).save(os.path.join(args.out, f"scene{sid:04d}_t{t}_GT.png"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H, W = args.image_size
fig, axes = plt.subplots(len(scene_ids), len(cols),
                         figsize=(len(cols) * W / 100.0, len(scene_ids) * H / 100.0))
axes = np.atleast_2d(axes)
for si, sid in enumerate(scene_ids):
    t = tgt_idx[si]
    for ci, c in enumerate(cols):
        ax = axes[si, ci]
        img = gt_cpu[si, t] if c == "GT" else renders[c][si, t]
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if si == 0:
            ax.set_title(c, fontsize=9)
        if c != "GT":
            ax.text(0.02, 0.03, f"{psnrs[c][si, t]:.2f} dB", transform=ax.transAxes,
                    fontsize=7, color="white", va="bottom", ha="left",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))
        if ci == 0:
            ax.set_ylabel(f"scene {sid}", fontsize=7)
plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.02, right=0.995, top=0.96, bottom=0.005)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(args.out, f"comparison.{ext}"), dpi=args.dpi, bbox_inches="tight")
print("saved ->", os.path.join(args.out, "comparison.{png,pdf}"))
