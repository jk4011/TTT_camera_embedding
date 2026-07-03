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
parser.add_argument("--bs", type=int, default=8)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

model_config = omegaconf.OmegaConf.load(args.config)
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
    eval_mode=True,
    num_input_views=n_in,
    num_target_views=n_tg,
    max_scenes=args.num_scenes,
)
loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=8)

lpips_model = lpips_lib.LPIPS(net="vgg").cuda().eval()

all_psnr, all_lpips = [], []
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
        lp = lpips_model(
            rendering.flatten(0, 1), target.flatten(0, 1), normalize=True
        ).reshape(rendering.size(0), -1).mean(dim=1)

        all_psnr.extend(psnr.cpu().tolist())
        all_lpips.extend(lp.cpu().tolist())

result = {
    "checkpoint": args.load,
    "num_scenes": len(all_psnr),
    "psnr": float(np.mean(all_psnr)),
    "lpips": float(np.mean(all_lpips)),
    "psnr_std_err": float(np.std(all_psnr) / np.sqrt(len(all_psnr))),
    "per_scene_psnr": all_psnr,
    "per_scene_lpips": all_lpips,
}
print(f"PSNR: {result['psnr']:.3f} +- {result['psnr_std_err']:.3f}  LPIPS: {result['lpips']:.4f}  ({result['num_scenes']} scenes)")

out_path = args.out or os.path.join(os.path.dirname(args.load), "eval.json")
with open(out_path, "w") as f:
    json.dump(result, f)
print(f"saved -> {out_path}")
