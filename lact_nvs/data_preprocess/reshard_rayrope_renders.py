"""Reshard the NEW vary-intrinsics renders (RayRoPE layout) into RE10K format.

Source: /NHNHOME/.../objaverse/renders/<uid>/{cameras.json, views/*.jpg}
        (24 views/object, PER-VIEW random FOV and distance -- the re10k format
        carries per-frame intrinsics, so the variation survives resharding.)
Output: /tmp/gobj_vi/{train,test}/<uid>.torch + {train,test}_index.json
Split:  last 500 sorted uids = test (matches the Q46 index builder).

Their cameras.json stores camera_to_world such that (stored @ blender2opencv)
is the opencv c2w (mirroring their parse_objaverse_camera); we invert to w2c.
Images are already 256x256 white-composited JPEGs: bytes stored verbatim.
"""
import json
import os
from multiprocessing import Pool

import numpy as np
import torch

SRC = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/objaverse/renders"
DST = "/tmp/gobj_vi"
B2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def convert(job):
    uid, split = job
    sdir = os.path.join(SRC, uid)
    out = os.path.join(DST, split, f"{uid}.torch")
    cams = json.load(open(os.path.join(sdir, "cameras.json")))["cameras"]
    if os.path.exists(out):
        return split, f"{uid}.torch", len(cams)
    rows, images = [], []
    for c in cams:
        w = h = 256.0
        fx, fy = c["intrinsics"]["focal_length"]
        cx, cy = c["intrinsics"]["principal_point"]
        c2w = np.eye(4)
        c2w[:3, :4] = np.array(c["extrinsics"]["camera_to_world"], dtype=np.float64)
        c2w = c2w @ B2CV
        w2c = np.linalg.inv(c2w)[:3].reshape(-1)
        rows.append(torch.tensor(
            [fx / w, fy / h, cx / w, cy / h, 0.0, 0.0, *w2c.tolist()],
            dtype=torch.float32))
        with open(os.path.join(sdir, "views", c["image_name"]), "rb") as f:
            images.append(torch.frombuffer(bytearray(f.read()), dtype=torch.uint8))
    scene = {"key": uid, "timestamps": torch.arange(len(cams)),
             "cameras": torch.stack(rows), "images": images}
    tmp = out + ".tmp"
    torch.save(scene, tmp)
    os.replace(tmp, out)
    return split, f"{uid}.torch", len(cams)


def main():
    uids = sorted(d for d in os.listdir(SRC)
                  if os.path.isfile(os.path.join(SRC, d, "cameras.json")))
    print(f"{len(uids)} rendered objects")
    test = set(uids[-500:])
    jobs = [(u, "test" if u in test else "train") for u in uids]
    for s in ("train", "test"):
        os.makedirs(os.path.join(DST, s), exist_ok=True)
    idx = {"train": [], "test": []}
    with Pool(48) as pool:
        for i, (split, fn, n) in enumerate(pool.imap_unordered(convert, jobs, chunksize=16)):
            idx[split].append({"file": fn, "num_frames": n})
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    for s in ("train", "test"):
        idx[s].sort(key=lambda e: e["file"])
        json.dump(idx[s], open(os.path.join(DST, f"{s}_index.json"), "w"))
        print(f"{s}: {len(idx[s])} scenes")
    print("RESHARD DONE")


if __name__ == "__main__":
    main()
