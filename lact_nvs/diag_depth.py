"""Depth diagnostic for the depth-predicted point-RoPE cells (DP program, 2026-09-02).

Records, per TTT layer, the depth t each dpt_* mode predicts (by wrapping
CamFastWeightGluMLPMultihead._dpt_depth) on held-out scenes and reports
  - s = log(t / t_c): mean / std over input and over target tokens (0 = point-RoPE's foot depth)
  - across-layer consistency: mean over tokens of the std of log t across layers
  - with --depth_dir (gObjaverse patch GT depth): mean |log t - log t_gt| for the predicted depth
    and for the foot depth t_c, and the Pearson correlation of log t with log t_gt,
    separately for input tokens (RGB available) and target tokens (pose only).
Usage mirrors eval.py:
  python diag_depth.py --load outputs/<exp>/model_0030000.pth --config config/dp_mlp_both.yaml
      [--data_path /tmp/gobj/test_index.json --min_frames 40 --depth_dir .../gobj_depth_patch/test]
      [--image_size 256 448] [--num_scenes 32] [--out outputs/<exp>/diag_depth.json]
"""
import argparse
import json
import os

import numpy as np
import omegaconf
import torch
from torch.utils.data import DataLoader

import lact_ttt_cam
from data_re10k import Re10KDataset
from model import LaCTLVSM

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--data_path", type=str, default="/tmp/re10k/test_index.json")
parser.add_argument("--num_scenes", type=int, default=32)
parser.add_argument("--num_input_views", type=int, default=8)
parser.add_argument("--num_target_views", type=int, default=4)
parser.add_argument("--image_size", nargs=2, type=int, default=[256, 256])
parser.add_argument("--window", type=int, default=128)
parser.add_argument("--min_frames", type=int, default=None)
parser.add_argument("--pose_norm_mode", type=str, default="mean", choices=["mean", "norecenter"])
parser.add_argument("--depth_dir", type=str, default=None)
parser.add_argument("--bs", type=int, default=8)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

model_config = omegaconf.OmegaConf.load(args.config)
model = LaCTLVSM(**model_config).cuda()
checkpoint = torch.load(args.load, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
model.eval()

# ---- wrap the depth function: record (layer_idx, t, t_c, t_gt, n_in) per call
records = []
_orig = lact_ttt_cam.CamFastWeightGluMLPMultihead._dpt_depth


def _rec(self, info, s):
    t = _orig(self, info, s)
    records.append((self.layer_idx, t.detach().float().cpu(), info["tok_tc"].detach().float().cpu(),
                    info["tok_t_gt"].detach().float().cpu() if "tok_t_gt" in info else None,
                    info["ttt_op_order"][0].end))
    return t


lact_ttt_cam.CamFastWeightGluMLPMultihead._dpt_depth = _rec

n_in, n_tg = args.num_input_views, args.num_target_views
dataset = Re10KDataset(
    args.data_path, num_views=n_in + n_tg, image_size=tuple(args.image_size),
    scene_pose_normalize=True, pose_norm_mode=args.pose_norm_mode, window=args.window,
    min_frames=args.min_frames, eval_mode=True, num_input_views=n_in, num_target_views=n_tg,
    max_scenes=args.num_scenes, depth_dir=args.depth_dir,
)
loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=4)

per_layer = {}   # layer -> dict of lists
per_scene_logt = []   # [n_layers, tokens] per batch, for across-layer consistency
with torch.no_grad():
    for data_dict in loader:
        data_dict = {k: v.cuda() for k, v in data_dict.items()}
        input_data_dict = {k: v[:, :n_in] for k, v in data_dict.items()}
        target_data_dict = {k: v[:, n_in:] for k, v in data_dict.items()}
        records.clear()
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):
            model(input_data_dict, target_data_dict)
        layer_logt = []
        for layer, t, tc, tgt, nin in records:
            d = per_layer.setdefault(layer, {"s_in": [], "s_tg": [], "e_in": [], "e_tg": [], "f_in": [], "f_tg": [],
                                             "lt_in": [], "lg_in": [], "lt_tg": [], "lg_tg": []})
            s = torch.log(t.clamp_min(1e-4) / tc.clamp_min(0.02))           # [b, L, 1]
            d["s_in"].append(s[:, :nin].flatten()); d["s_tg"].append(s[:, nin:].flatten())
            layer_logt.append(torch.log(t.clamp_min(1e-4)).flatten())
            if tgt is not None:
                lt, lg, lc = torch.log(t.clamp_min(1e-4)), torch.log(tgt.clamp_min(1e-4)), torch.log(tc.clamp_min(0.02))
                for tag, sl in (("in", slice(0, nin)), ("tg", slice(nin, None))):
                    ok = (tgt[:, sl] > 0).flatten()
                    d["e_" + tag].append((lt[:, sl] - lg[:, sl]).abs().flatten()[ok])
                    d["f_" + tag].append((lc[:, sl] - lg[:, sl]).abs().flatten()[ok])
                    d["lt_" + tag].append(lt[:, sl].flatten()[ok]); d["lg_" + tag].append(lg[:, sl].flatten()[ok])
        if layer_logt:
            per_scene_logt.append(torch.stack(layer_logt).std(0))            # std across layers, per token


def _corr(a, b):
    if len(a) < 3:
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


out = {"checkpoint": args.load, "num_scenes": len(dataset), "layers": {}}
hdr = "layer | s_in mean/std | s_tg mean/std"
if args.depth_dir:
    hdr += " | |logerr| in: pred / foot / corr | tg: pred / foot / corr"
print(hdr)
for layer in sorted(per_layer):
    d = {k: (torch.cat(v) if v else torch.zeros(0)) for k, v in per_layer[layer].items()}
    row = {"s_in_mean": float(d["s_in"].mean()), "s_in_std": float(d["s_in"].std()),
           "s_tg_mean": float(d["s_tg"].mean()), "s_tg_std": float(d["s_tg"].std())}
    line = f"{layer:5d} | {row['s_in_mean']:+.3f} / {row['s_in_std']:.3f} | {row['s_tg_mean']:+.3f} / {row['s_tg_std']:.3f}"
    if args.depth_dir and len(d["e_in"]):
        row.update(err_in=float(d["e_in"].mean()), foot_err_in=float(d["f_in"].mean()), corr_in=_corr(d["lt_in"], d["lg_in"]),
                   err_tg=float(d["e_tg"].mean()), foot_err_tg=float(d["f_tg"].mean()), corr_tg=_corr(d["lt_tg"], d["lg_tg"]))
        line += (f" | {row['err_in']:.3f} / {row['foot_err_in']:.3f} / {row['corr_in']:+.2f}"
                 f" | {row['err_tg']:.3f} / {row['foot_err_tg']:.3f} / {row['corr_tg']:+.2f}")
    print(line)
    out["layers"][str(layer)] = row
if per_scene_logt:
    out["cross_layer_std_logt"] = float(torch.cat(per_scene_logt).mean())
    print(f"across-layer std of log t (mean over tokens): {out['cross_layer_std_logt']:.3f}")
out_path = args.out or os.path.join(os.path.dirname(args.load), "diag_depth.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
print(f"saved -> {out_path}")
