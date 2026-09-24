"""Stage 1 of the ccv camera-accuracy metric: frames on disk + ground-truth poses.

`eval_ccv_generate.py` saves one mp4 per held-out pair and scores pixels against the
GT target-camera video. RotErr / TransErr / CamMC instead ask whether the camera the
model actually rendered is the camera it was told to render, which means estimating
poses FROM the generated video with structure-from-motion and comparing them to the
conditioning trajectory.

COLMAP lives in its own environment (`envs/sfm`, so a dependency bump can never touch a
running training), and it reads images from disk, so this stage runs in the training
environment and writes what stage 2 needs:

  <out>/<tag>/gen/frame%04d.jpg   the generated video
  <out>/<tag>/gt/frame%04d.jpg    the GT target-camera video (stage 2 scores this too:
                                  SfM on real frames is the pipeline's own sanity check
                                  and the floor the generated numbers are read against)
  <out>/pairs.json                per pair: the conditioning c2w_tgt [21, 4, 4], K, and
                                  the frame indices those poses belong to (0, 4, ..., 80)

Poses come from the dataset loader itself (same canonicalisation the model was
conditioned on: joint mean-pose normalisation over the src and tgt cameras).

Usage (from lact_ar_video/minVid, PYTHONPATH=<repo>/lact_ar_video):
  python eval_ccv_campose_prep.py --gen_dir ../outputs/eval/gen_ccv_base_13999 \
      --out ../outputs/eval/campose_ccv_base_13999
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from eval_ccv_common import (
    DEFAULT_PAIRS_JSON, load_config, load_or_build_pairs, make_pair_dataset,
)


def tag_of(rec):
    """The tag eval_ccv_generate.py builds its file names from."""
    return (f"pair{rec['index']:03d}_{rec['relpath'].replace(os.sep, '_')}"
            f"_cam{rec['src_cam']:02d}to{rec['tgt_cam']:02d}")


def save_frames(frames, out_dir, quality=95):
    """frames: [F, C, H, W] float in [0, 1]."""
    os.makedirs(out_dir, exist_ok=True)
    arr = (frames.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    for i, im in enumerate(arr):
        Image.fromarray(im).save(os.path.join(out_dir, f"frame{i:04d}.jpg"), quality=quality)
    return arr.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default=None, help="eval_ccv_generate.py output dir")
    ap.add_argument("--from_dataset", type=int, default=0,
                    help="no generated videos yet: take the first N pairs of the dataset index "
                         "and write only their GT frames (validates the metric end to end, and "
                         "gives the floor stage 2 reads the generated numbers against)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_pairs", type=int, default=0, help="0 = all pairs in the dir")
    ap.add_argument("--config", default="configs/ar/abl_ccv_base.yaml",
                    help="only its dataset_train params are used")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS_JSON)
    ap.add_argument("--skip_gt", action="store_true")
    args = ap.parse_args()

    assert args.gen_dir or args.from_dataset, "pass --gen_dir or --from_dataset N"
    recs = None
    if args.gen_dir:
        with open(os.path.join(args.gen_dir, "metrics.json")) as f:
            recs = json.load(f)["per_pair"]
        if args.n_pairs:
            recs = recs[:args.n_pairs]
    os.makedirs(args.out, exist_ok=True)

    # the evaluated pairs are the HELD-OUT list, which is disjoint from the training pair
    # index -- building the dataset from that index would not contain a single one of them
    cfg = load_config(args.config)
    pairs = load_or_build_pairs(args.pairs, cfg)["pairs"]
    ds = make_pair_dataset(cfg, pairs)
    by_key = {(p["relpath"], p["src_cam"], p["tgt_cam"]): i for i, p in enumerate(pairs)}
    if recs is None:
        recs = [{"index": i, "relpath": p["relpath"], "src_cam": p["src_cam"],
                 "tgt_cam": p["tgt_cam"]} for i, p in enumerate(pairs[:args.from_dataset])]

    out = {"pairs": [], "pose_frame_ids": list(range(0, ds.num_frames, ds.pose_stride))}
    for rec in recs:
        tag = tag_of(rec)
        key = (rec["relpath"], rec["src_cam"], rec["tgt_cam"])
        assert key in by_key, f"pair {key} is not in the held-out list"
        item = ds[by_key[key]]

        n_gen = 0
        gen_mp4 = os.path.join(args.gen_dir, f"{tag}_gen.mp4") if args.gen_dir else None
        if gen_mp4 and os.path.isfile(gen_mp4):
            import imageio.v3 as iio
            vid = torch.from_numpy(np.asarray(iio.imread(gen_mp4))).float().div_(255.0)
            n_gen = save_frames(vid.permute(0, 3, 1, 2), os.path.join(args.out, tag, "gen"))
        elif gen_mp4:
            print(f"[prep] MISSING {gen_mp4}", flush=True)
        if not args.skip_gt:
            save_frames(item["frames_tgt"].permute(1, 0, 2, 3), os.path.join(args.out, tag, "gt"))

        out["pairs"].append({
            "tag": tag, "index": rec["index"], "relpath": rec["relpath"],
            "src_cam": rec["src_cam"], "tgt_cam": rec["tgt_cam"],
            "n_gen_frames": n_gen,
            "K": item["K"].tolist(),
            "c2w_tgt": item["c2w_tgt"].tolist(),
        })
        print(f"[prep] {tag}: {n_gen} generated frames", flush=True)

    with open(os.path.join(args.out, "pairs.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[prep] wrote {len(out['pairs'])} pairs -> {args.out}/pairs.json", flush=True)


if __name__ == "__main__":
    main()
