"""Reshard DL3DV (opencv_cameras.json + PNG frames) into RE10K per-scene .torch format.

Why this exists: the fastest way to train LaCT-LVSM on DL3DV is to make DL3DV look
exactly like the RE10K working copy -- then train.py (--dataset re10k), eval.py and
every launcher work unchanged, including the eval-mode view selection (8 uniform
inputs / 4 midpoint targets) that defines the standard protocol. The alternative
(NVSDataset) has no eval_mode, so eval.py would have needed a fork.

Output format (identical to reshard_re10k.py):
  <odir>/<split>/<scene>.torch : {key, timestamps [N], cameras [N,18], images: list of
                                  N jpeg-byte uint8 tensors}
  cameras row: [fx fy cx cy 0 0, w2c(3x4) flattened], intrinsics NORMALIZED by image
  size (DL3DV stores them in pixels -- divide by w/h here).
  <odir>/<split>_index.json    : [{"file", "num_frames"}]

Images are resized to cover 256x256 (960x540 -> 455x256) and stored as JPEG q95.
Training decodes with decode_resize_crop, which resizes to cover the target and
center-crops, so pre-sizing to the cover size makes that step a no-op. This is one
more resampling than RE10K frames go through (they are stored at native 640x360);
a wash for a four-arm contrast where every arm reads the same bytes.

Usage:
  python reshard_dl3dv.py --src .../dl3dv_undistorted_960/train --odir /tmp/dl3dv/train \
      --index /tmp/dl3dv/train_index.json --workers 64 [--short_side 256]
"""
import argparse
import io
import json
import os
from multiprocessing import Pool

import torch

# a scene is a directory with opencv_cameras.json + images_undistort/*.png
CAM_JSON = "opencv_cameras.json"


def process_scene(job):
    scene_dir, out_dir, short_side = job
    key = os.path.basename(scene_dir.rstrip("/"))
    out_path = os.path.join(out_dir, f"{key}.torch")
    meta = json.load(open(os.path.join(scene_dir, CAM_JSON)))
    frames = meta["frames"]
    if os.path.exists(out_path):
        return {"file": f"{key}.torch", "num_frames": len(frames)}

    from PIL import Image  # import inside the worker

    cameras, images = [], []
    for fr in frames:
        w, h = fr["w"], fr["h"]
        # normalize intrinsics by image size, like the RE10K chunks
        cam = [fr["fx"] / w, fr["fy"] / h, fr["cx"] / w, fr["cy"] / h, 0.0, 0.0]
        w2c = torch.tensor(fr["w2c"], dtype=torch.float32)[:3].reshape(-1)
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
    os.replace(tmp, out_path)  # atomic: an interrupted reshard never leaves a bad file
    return {"file": f"{key}.torch", "num_frames": len(frames)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--odir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--short_side", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.odir, exist_ok=True)
    scenes = sorted(
        os.path.join(args.src, d)
        for d in os.listdir(args.src)
        if os.path.isfile(os.path.join(args.src, d, CAM_JSON))
    )
    print(f"{len(scenes)} scenes in {args.src}")

    # PIL is single-threaded here; cap BLAS threads so workers*threads != carnage
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    entries = []
    with Pool(args.workers) as pool:
        jobs = [(s, args.odir, args.short_side) for s in scenes]
        for i, e in enumerate(pool.imap_unordered(process_scene, jobs, chunksize=4)):
            entries.append(e)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(scenes)} scenes", flush=True)

    entries.sort(key=lambda e: e["file"])
    tmp = args.index + ".tmp"
    json.dump(entries, open(tmp, "w"))
    os.replace(tmp, args.index)
    print(f"DONE: {len(entries)} scenes, index -> {args.index}")


if __name__ == "__main__":
    main()
