"""Input-view sweep with FIXED target views (2026-09-25, Objaverse check for Fig. 5).

The standard eval selection (data_re10k._select_indices) moves the 4 targets off the inputs, so on the
40-view G-Objaverse orbits the target set changes with the input count (V=8: 5,15,24,34; V=16:
6,15,24,35; V=32: 7,17,27,37), and one target is always a duplicate of an input (24 == 0, 27 == 39:
same camera, same image). Here the inputs are the standard uniform ones, but the targets are fixed to
frames that are never an input at V in {4,8,16,32} and have no duplicate: 7, 12 (upper ring), 32, 37
(lower ring). Per-scene PSNR as in eval.py (PSNR of the mean MSE over the targets).

  python eval_fixed_targets.py --config C --load CKPT --views 4 8 16 32 --out out.json
"""
import argparse
import json

import numpy as np
import omegaconf
import torch
from torch.utils.data import DataLoader

import data_re10k
from data_re10k import Re10KDataset
from model import LaCTLVSM

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--load", required=True)
p.add_argument("--data_path", default="/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/reshard/gobj/test_index.json")
p.add_argument("--num_scenes", type=int, default=500)
p.add_argument("--min_frames", type=int, default=40)
p.add_argument("--targets", type=str, default="7,12,32,37")
p.add_argument("--views", type=int, nargs="+", default=[4, 8, 16, 32])
p.add_argument("--out", required=True)
args = p.parse_args()
T = [int(x) for x in args.targets.split(",")]

model = LaCTLVSM(**omegaconf.OmegaConf.load(args.config)).cuda()
sd = torch.load(args.load, map_location="cpu", weights_only=False)
model.load_state_dict(sd["model"] if "model" in sd else sd); model.eval()

res = {"targets": T, "psnr": {}, "per_view_psnr": {}}
for V in args.views:
    orig = Re10KDataset._select_indices

    def sel(self, num_frames, V=V):
        assert num_frames == 40, num_frames
        std = orig(self, num_frames)[:V]                     # the standard uniform inputs
        assert not (set(std) & set(T)), (V, std, T)
        return std + T
    Re10KDataset._select_indices = sel
    ds = Re10KDataset(args.data_path, num_views=V + len(T), image_size=(256, 256), scene_pose_normalize=True,
                      pose_norm_mode="mean", window=128, min_frames=args.min_frames, eval_mode=True,
                      num_input_views=V, num_target_views=len(T), max_scenes=args.num_scenes)
    bs = 8 if V <= 8 else (4 if V <= 16 else 2)
    ps, pv = [], []
    with torch.no_grad():
        for d in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8):
            d = {k: v.cuda() for k, v in d.items()}
            inp = {k: v[:, :V] for k, v in d.items()}; tgt = {k: v[:, V:] for k, v in d.items()}
            with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                r = model(inp, tgt)
            r = r.float().clamp(0, 1); g = tgt["image"].float()
            mse_v = ((r - g) ** 2).flatten(2).mean(2)            # [b, 4]
            ps.extend((-10 * torch.log10(mse_v.mean(1))).tolist())
            pv.extend((-10 * torch.log10(mse_v)).tolist())
    Re10KDataset._select_indices = orig
    res["psnr"][V] = float(np.mean(ps)); res["per_view_psnr"][V] = np.mean(pv, 0).tolist()
    print(f"V={V:2d} n={len(ps)} PSNR {res['psnr'][V]:.3f}  per target " +
          " ".join(f"{t}:{x:.2f}" for t, x in zip(T, res["per_view_psnr"][V])), flush=True)
json.dump(res, open(args.out, "w"), indent=1)
