"""Reshard gObjaverse (WAI format) into the RE10K per-scene .torch format.

Why this exists: same reason as reshard_dl3dv.py -- making the data look exactly like
the RE10K working copy means train.py (--dataset re10k), eval.py and every launcher
work unchanged, including the eval-mode view selection (8 uniform inputs / 4 midpoint
targets) that defines the standard protocol.

WHY gObjaverse. It is the OBJECT-level end of the camera-baseline axis this line of
work is testing. Measured with one method across all three (angle between camera
forward axes, over the views the loader actually serves):

    RE10K ~7 deg   ->   DL3DV 34.5 deg   ->   gObjaverse 89.3 deg

Cameras sit on a sphere (radius ~1.5) around the object, so the max pairwise angle is
a full 180 deg. LaCT's own object-level checkpoint was trained on Objaverse renders
(32 views/object, LVSM/GS-LRM settings) which were never released; gObjaverse is the
public pre-rendered stand-in, at 40 views/object.

SCOPE WARNING to state in any write-up: object-level changes more than the baseline --
background, object-centric normalization and bounded scene scale all move too. This is
NOT a clean single-variable extension of the DL3DV grid; it is coverage of the setting
LaCT itself used.

Input (WAI): <src>/<scene>/scene_meta.json + images/*.jpg, with per-frame
`transform_matrix` = 4x4 CAMERA-TO-WORLD in the opencv convention. Verified, not
assumed: every camera's +Z axis points at the origin (mean dot(fwd, center_hat)
= -1.0000; the inverse gives -0.217). The RE10K format wants w2c, so we invert.

Output (identical to reshard_dl3dv.py / reshard_re10k.py):
  <odir>/<scene>.torch : {key, timestamps [N], cameras [N,18], images: list of N
                          jpeg-byte uint8 tensors}
  cameras row: [fx fy cx cy 0 0, w2c(3x4) flattened], intrinsics NORMALIZED by image
  size (WAI stores them in pixels -- divide by w/h here).
  <index>              : [{"file", "num_frames"}]

Split: gObjaverse ships no official train/test split, so one is derived
deterministically from the sorted scene list -- the LAST --n_test scenes are test,
the rest train. Sorted-and-fixed so both splits are reproducible and provably
disjoint, and so re-running never silently reshuffles them.

Usage:
  python reshard_gobjaverse.py --src .../gobjaverse_wai --odir /tmp/gobj/train \
      --index /tmp/gobj/train_index.json --split train --workers 64
"""
import argparse
import io
import json
import os
from multiprocessing import Pool

import numpy as np
import torch

META = "scene_meta.json"


def process_scene(job):
    scene_dir, out_dir, short_side = job
    key = os.path.basename(scene_dir.rstrip("/"))
    out_path = os.path.join(out_dir, f"{key}.torch")
    meta = json.load(open(os.path.join(scene_dir, META)))
    frames = meta["frames"]
    if os.path.exists(out_path):
        return {"file": f"{key}.torch", "num_frames": len(frames)}

    from PIL import Image  # import inside the worker

    cameras, images = [], []
    for fr in frames:
        w, h = fr["w"], fr["h"]
        cam = [fr["fl_x"] / w, fr["fl_y"] / h, fr["cx"] / w, fr["cy"] / h, 0.0, 0.0]
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64).reshape(4, 4)
        w2c = np.linalg.inv(c2w)[:3].reshape(-1)
        cameras.append(torch.tensor(cam + w2c.tolist(), dtype=torch.float32))

        img = Image.open(os.path.join(scene_dir, fr["file_path"])).convert("RGB")
        scale = short_side / min(img.size[0], img.size[1])
        img = img.resize(
            (max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale))),
            Image.LANCZOS,
        )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        images.append(torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8))

    scene = {
        "key": key,
        "timestamps": torch.arange(len(frames)),
        "cameras": torch.stack(cameras),
        "images": images,
    }
    tmp = out_path + ".tmp"
    torch.save(scene, tmp)
    os.replace(tmp, out_path)  # atomic: an interrupted reshard leaves no bad file
    return {"file": f"{key}.torch", "num_frames": len(frames)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--odir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--max_scenes", type=int, default=None,
                    help="cap the TRAIN split (test is always the full holdout)")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--short_side", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.odir, exist_ok=True)
    all_scenes = sorted(
        d for d in os.listdir(args.src)
        if os.path.isfile(os.path.join(args.src, d, META))
    )
    test, train = all_scenes[-args.n_test:], all_scenes[:-args.n_test]
    assert not (set(test) & set(train)), "train/test overlap"
    picked = test if args.split == "test" else train
    if args.split == "train" and args.max_scenes:
        picked = picked[:args.max_scenes]
    print(f"{len(all_scenes)} scenes total -> {args.split}: {len(picked)} "
          f"(holdout {args.n_test}, disjoint)")

    os.environ.setdefault("OMP_NUM_THREADS", "1")

    entries = []
    with Pool(args.workers) as pool:
        jobs = [(os.path.join(args.src, s), args.odir, args.short_side)
                for s in picked]
        for i, e in enumerate(pool.imap_unordered(process_scene, jobs, chunksize=4)):
            entries.append(e)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(picked)} scenes", flush=True)

    entries.sort(key=lambda e: e["file"])
    tmp = args.index + ".tmp"
    json.dump(entries, open(tmp, "w"))
    os.replace(tmp, args.index)
    print(f"DONE: {len(entries)} scenes, index -> {args.index}")


if __name__ == "__main__":
    main()
