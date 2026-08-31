"""Patch-level GT depth side files for gObjaverse (oracle 3D-point addressing diagnostics).

For every scene in the WAI source, reads the 40 EXR z-depth maps (512x512, 'RGB' channel,
z-depth in the camera frame -- verified against cross-view reprojection: median abs error
0.0008 for z-depth vs 0.037 for ray-distance) and pools them onto the 16x16 patch grid the
model uses at 256x256 / patch 16 (each patch = 32x32 source pixels):

  t[f, py, px]     = median z over the VALID (z > 0) pixels of the patch, converted to the
                     ray parameter of the patch-centre ray (x = o + t d, |d| = 1):
                     t = z * sqrt(1 + x^2 + y^2), (x, y) = normalised pixel coords.
                     0 where the patch has no valid pixel (background).
  valid[f, py, px] = fraction of valid pixels in the patch.

Values are in the RAW metric frame of scene_meta.json; the loader divides t by the scene
scale that normalize_with_mean_pose applies to the camera centres (rays are unit-length, so
t scales exactly like translations).

Output: <odir>/<scene>.pt = {"t": float32 [N,16,16], "valid": float16 [N,16,16]}, one per
scene, same scene keys as reshard_gobjaverse.py. Split logic identical to that script
(last --n_test sorted keys = test) so the side files pair 1:1 with /tmp/gobj/{train,test}.

Usage:
  python gobj_patch_depth.py --src .../gobjaverse_wai --odir .../gobj_depth_patch --workers 56
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

META = "scene_meta.json"
GRID = 16


def _one(args):
    src, scene, odir = args
    out = os.path.join(odir, scene + ".pt")
    if os.path.exists(out):
        return scene, True
    import OpenEXR
    import torch
    meta = json.load(open(os.path.join(src, scene, META)))
    frames = meta["frames"]
    n = len(frames)
    t_all = np.zeros((n, GRID, GRID), np.float32)
    v_all = np.zeros((n, GRID, GRID), np.float32)
    for fi, fr in enumerate(frames):
        p = os.path.join(src, scene, fr["depth"])
        try:
            with OpenEXR.File(p) as f:
                ch = f.channels()
                arr = ch["RGB"].pixels if "RGB" in ch else next(iter(ch.values())).pixels
                z = arr[..., 0].astype(np.float32) if arr.ndim == 3 else arr.astype(np.float32)
        except Exception:
            continue
        H, W = z.shape
        fx, fy, cx, cy = fr["fl_x"], fr["fl_y"], fr["cx"], fr["cy"]
        bh, bw = H // GRID, W // GRID
        zb = z[: bh * GRID, : bw * GRID].reshape(GRID, bh, GRID, bw).transpose(0, 2, 1, 3)
        zb = zb.reshape(GRID, GRID, bh * bw)
        valid = (zb > 0) & np.isfinite(zb)
        vfrac = valid.mean(-1)
        zmed = np.zeros((GRID, GRID), np.float32)
        for py in range(GRID):
            for px in range(GRID):
                m = valid[py, px]
                if m.any():
                    zmed[py, px] = np.median(zb[py, px][m])
        us = (np.arange(GRID) * bw + bw / 2.0 - cx) / fx
        vs = (np.arange(GRID) * bh + bh / 2.0 - cy) / fy
        xx, yy = np.meshgrid(us, vs, indexing="xy")
        t_all[fi] = zmed * np.sqrt(1.0 + xx**2 + yy**2) * (vfrac > 0)
        v_all[fi] = vfrac
    torch.save({"t": torch.from_numpy(t_all), "valid": torch.from_numpy(v_all).half()}, out)
    return scene, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--odir", required=True)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--workers", type=int, default=56)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    scenes = sorted(d for d in os.listdir(args.src)
                    if os.path.isfile(os.path.join(args.src, d, META)))
    test, train = scenes[-args.n_test:], scenes[:-args.n_test]
    jobs = []
    for split, lst in (("test", test), ("train", train)):
        if args.split in (split, "both"):
            od = os.path.join(args.odir, split)
            os.makedirs(od, exist_ok=True)
            jobs += [(args.src, s, od) for s in lst]
    print(f"{len(jobs)} scenes", flush=True)
    done = 0
    with Pool(args.workers) as pool:
        for _ in pool.imap_unordered(_one, jobs, chunksize=8):
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    print("DEPTH_DONE", flush=True)


if __name__ == "__main__":
    main()
