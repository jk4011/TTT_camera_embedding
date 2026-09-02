"""Per-scene angular spread of the INPUT views' optical axes (mean pairwise angle between the camera
forward vectors), on the eval protocol of each dataset -- the statistic a geometry-adaptive gate on the
direction half of pdir would use. Usage: python diag_baseline_angle.py --data_path ... [--min_frames N]
[--image_size 256 448] [--num_scenes 64]"""
import argparse
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_re10k import Re10KDataset

p = argparse.ArgumentParser()
p.add_argument("--data_path", default="/tmp/re10k/test_index.json")
p.add_argument("--num_scenes", type=int, default=64)
p.add_argument("--num_input_views", type=int, default=8)
p.add_argument("--num_target_views", type=int, default=4)
p.add_argument("--image_size", nargs=2, type=int, default=[256, 256])
p.add_argument("--window", type=int, default=128)
p.add_argument("--min_frames", type=int, default=None)
a = p.parse_args()
ds = Re10KDataset(a.data_path, num_views=a.num_input_views + a.num_target_views, image_size=tuple(a.image_size),
                  scene_pose_normalize=True, window=a.window, min_frames=a.min_frames, eval_mode=True,
                  num_input_views=a.num_input_views, num_target_views=a.num_target_views, max_scenes=a.num_scenes)
angles = []
for d in DataLoader(ds, batch_size=8, num_workers=4):
    c2w = d["c2w"][:, :a.num_input_views].float()               # [b, V, 4, 4]
    f = c2w[..., :3, 2]                                          # forward axis (camera +z), world frame
    f = f / f.norm(dim=-1, keepdim=True)
    cs = (f @ f.transpose(1, 2)).clamp(-1, 1)                    # [b, V, V]
    ang = torch.acos(cs) * 180 / math.pi
    V = f.shape[1]
    off = ~torch.eye(V, dtype=torch.bool)
    angles.extend(ang[:, off].mean(1).tolist())
angles = np.array(angles)
print(f"{a.data_path}: n={len(angles)} scenes; mean pairwise forward-axis angle: "
      f"median {np.median(angles):.1f} deg, p10 {np.percentile(angles, 10):.1f}, p90 {np.percentile(angles, 90):.1f}, "
      f"min {angles.min():.1f}, max {angles.max():.1f}")
