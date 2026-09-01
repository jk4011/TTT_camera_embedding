"""Deterministic evaluation on held-out RE10K test scenes.

Protocol: centered window of up to --window frames; --num_input_views inputs
uniformly spaced; --num_target_views targets at window midpoints (see
Re10KDataset eval_mode). Reports mean PSNR / LPIPS over --num_scenes scenes.
"""
import argparse
import json
import os

import lpips as lpips_lib
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_re10k import Re10KDataset
from model import LaCTLVSM

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--data_path", type=str, default="/tmp/re10k/test_index.json")
parser.add_argument("--num_scenes", type=int, default=256)
parser.add_argument("--num_input_views", type=int, default=8)
parser.add_argument("--num_target_views", type=int, default=4)
parser.add_argument("--image_size", nargs=2, type=int, default=[256, 256])
parser.add_argument("--window", type=int, default=128)
# Re10KDataset drops scenes with fewer than num_views*3 frames. That default scales
# with the view count, so a VIEW SWEEP silently changes the scene set as it goes --
# and on a fixed-length dataset it can empty the set entirely (gObjaverse has 40
# frames/scene, so 16 inputs => threshold 60 => 0 scenes, PSNR nan). Pin it to keep
# one scene set across the whole sweep. None keeps the old behaviour.
parser.add_argument("--min_frames", type=int, default=None)
parser.add_argument("--depth_dir", type=str, default=None,
                    help="patch-grid GT depth side files (oracle diagnostics only)")
parser.add_argument("--bs", type=int, default=8)
parser.add_argument("--ttt_num_chunks", type=int, default=1,
                    help="Split the input-token fast-weight update into n sequential "
                         "chunks at EVAL time. The method is derived for one update "
                         "step; this tests that the learned phases survive chunking. "
                         "1 reproduces the trained setting.")
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

model_config = omegaconf.OmegaConf.load(args.config)
if args.ttt_num_chunks > 1:
    model_config["ttt_num_chunks"] = args.ttt_num_chunks
model = LaCTLVSM(**model_config).cuda()

checkpoint = torch.load(args.load, map_location="cpu", weights_only=False)
state = checkpoint["model"] if "model" in checkpoint else checkpoint
model.load_state_dict(state)
model.eval()

n_in, n_tg = args.num_input_views, args.num_target_views
dataset = Re10KDataset(
    args.data_path,
    num_views=n_in + n_tg,
    image_size=tuple(args.image_size),
    scene_pose_normalize=True,
    window=args.window,
    min_frames=args.min_frames,
    eval_mode=True,
    num_input_views=n_in,
    num_target_views=n_tg,
    max_scenes=args.num_scenes,
    depth_dir=args.depth_dir,
)
loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=8)

lpips_model = lpips_lib.LPIPS(net="vgg").cuda().eval()


def ssim_fn(a, b):
    """Standard SSIM (11x11 Gaussian window, sigma 1.5, K1/K2 = 0.01/0.03,
    data_range 1), per-channel then averaged: the skimage/torchmetrics convention
    the tttLRM numbers use. a, b: [N, C, H, W] in [0, 1]; returns [N]."""
    coords = torch.arange(11, dtype=torch.float32, device=a.device) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = (g / g.sum()).outer(g / g.sum())
    w = g.expand(a.size(1), 1, 11, 11)
    mu_a = F.conv2d(a, w, groups=a.size(1))
    mu_b = F.conv2d(b, w, groups=a.size(1))
    var_a = F.conv2d(a * a, w, groups=a.size(1)) - mu_a**2
    var_b = F.conv2d(b * b, w, groups=a.size(1)) - mu_b**2
    cov = F.conv2d(a * b, w, groups=a.size(1)) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return s.flatten(1).mean(dim=1)


all_psnr, all_lpips, all_ssim = [], [], []
all_psnr_pv = []   # per-scene, per-target-view PSNR (P2 diagnostic: inner vs outer targets)
with torch.no_grad():
    for data_dict in loader:
        data_dict = {k: v.cuda() for k, v in data_dict.items()}
        input_data_dict = {k: v[:, :n_in] for k, v in data_dict.items()}
        target_data_dict = {k: v[:, n_in:] for k, v in data_dict.items()}

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):
            rendering = model(input_data_dict, target_data_dict)
        rendering = rendering.float().clamp(0, 1)
        target = target_data_dict["image"].float()

        # Per-scene PSNR over its target views
        mse = ((rendering - target) ** 2).flatten(1).mean(dim=1)
        psnr = -10.0 * torch.log10(mse)
        mse_pv = ((rendering - target) ** 2).flatten(2).mean(dim=2)          # [b, n_tgt]
        all_psnr_pv.extend((-10.0 * torch.log10(mse_pv)).cpu().tolist())
        lp = lpips_model(
            rendering.flatten(0, 1), target.flatten(0, 1), normalize=True
        ).reshape(rendering.size(0), -1).mean(dim=1)
        ss = ssim_fn(
            rendering.flatten(0, 1), target.flatten(0, 1)
        ).reshape(rendering.size(0), -1).mean(dim=1)

        all_psnr.extend(psnr.cpu().tolist())
        all_lpips.extend(lp.cpu().tolist())
        all_ssim.extend(ss.cpu().tolist())

result = {
    "checkpoint": args.load,
    "num_scenes": len(all_psnr),
    "psnr": float(np.mean(all_psnr)),
    "lpips": float(np.mean(all_lpips)),
    "ssim": float(np.mean(all_ssim)),
    "psnr_std_err": float(np.std(all_psnr) / np.sqrt(len(all_psnr))),
    "per_scene_psnr": all_psnr,
    "per_scene_lpips": all_lpips,
    "per_scene_ssim": all_ssim,
    "per_view_psnr": [float(x) for x in np.mean(np.array(all_psnr_pv), axis=0)],
    "per_scene_per_view_psnr": all_psnr_pv,
}
print(f"PSNR: {result['psnr']:.3f} +- {result['psnr_std_err']:.3f}  SSIM: {result['ssim']:.4f}  LPIPS: {result['lpips']:.4f}  ({result['num_scenes']} scenes)")

out_path = args.out or os.path.join(os.path.dirname(args.load), "eval.json")
with open(out_path, "w") as f:
    json.dump(result, f)
print(f"saved -> {out_path}")
