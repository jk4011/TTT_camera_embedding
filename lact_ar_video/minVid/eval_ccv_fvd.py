"""tab:ccv FVD (2026-09-25): I3D FVD of each generated set against the real target videos.

Real side: the ground-truth target videos of the same held-out pairs, taken from the right third of
the [src | gen | gt] side-by-side files eval_ccv_generate.py wrote. Those GT frames went through the
same preprocessing and the same single video encode as the *_gen.mp4 files, so both sides are treated
alike (the dataset's own cam*.mp4 are 1280x1280 and were never seen at that size).
Preprocessing follows cd-fvd's video loader (shorter side to 224, bilinear with antialias, centre
crop 224), but every video contributes all of its frames as non-overlapping 16-frame clips
(81 frames -> 5 clips, frames 0-79) instead of only the first 16.

  .venv_llm/bin/python eval_ccv_fvd.py --gt_from ../outputs/eval/gen_ccv_base_re_13999 \
      --fake_dirs ../outputs/eval/gen_ccv_base_re_13999 ../outputs/eval/gen_ccv_capet_re_13999 \
      --out ../outputs/eval/fvd_ccv_13999.json
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

# cd-fvd expects the old scipy sqrtm(..., disp=False) tuple API
import scipy.linalg as sla
_orig_sqrtm = sla.sqrtm
def _sqrtm(A, disp=None, **kw):
    r = _orig_sqrtm(A)
    return (r, 0.0) if disp is not None else r
sla.sqrtm = _sqrtm
from cdfvd import fvd  # noqa: E402


def read(path, x0=None, width=None):
    v = VideoReader(path, ctx=cpu(0))
    a = v.get_batch(range(len(v))).asnumpy()                     # [T, H, W, 3] uint8
    if x0 is not None:
        a = a[:, :, x0:x0 + width]
    return a


def clips(a, res=224, L=16):
    t = torch.from_numpy(a).permute(0, 3, 1, 2).float()          # [T, 3, H, W]
    h, w = t.shape[-2:]
    s = res / min(h, w)
    size = (res, math.ceil(w * s)) if h < w else (math.ceil(h * s), res)
    t = F.interpolate(t, size=size, mode="bilinear", align_corners=False, antialias=True)
    h, w = t.shape[-2:]
    t = t[:, :, (h - res) // 2:(h - res) // 2 + res, (w - res) // 2:(w - res) // 2 + res]
    t = t.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()   # [T, res, res, 3]
    n = t.shape[0] // L
    return [t[i * L:(i + 1) * L] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_from", required=True, help="a generation dir whose *_src_gen_gt.mp4 hold the GT")
    ap.add_argument("--fake_dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None, help="where the .npy clip arrays go (default: next to --out)")
    args = ap.parse_args()
    tmp = args.tmp or os.path.splitext(args.out)[0] + "_npy"
    os.makedirs(tmp, exist_ok=True)

    tags = lambda d: {os.path.basename(f)[:-len("_gen.mp4")] for f in glob.glob(os.path.join(d, "*_gen.mp4"))}
    common = sorted(set.intersection(*[tags(d) for d in args.fake_dirs + [args.gt_from]]))
    print(f"[fvd] {len(common)} pairs common to all sets", flush=True)

    def dump(name, videos):
        arr = np.stack([c for v in videos for c in clips(v)])
        path = os.path.join(tmp, f"{name}.npy"); np.save(path, arr)
        print(f"[fvd] {name}: {arr.shape[0]} clips of {arr.shape[1:]}", flush=True)
        return path

    gt_videos = []
    for tg in common:
        g = read(os.path.join(args.gt_from, f"{tg}_gen.mp4"))
        sxs = read(os.path.join(args.gt_from, f"{tg}_src_gen_gt.mp4"))
        w = g.shape[2]
        assert sxs.shape[2] == 3 * w, (sxs.shape, g.shape)
        gt_videos.append(sxs[:, :, 2 * w:3 * w])
    real = dump("real_gt", gt_videos)

    ev = fvd.cdfvd("i3d", n_real="full", n_fake="full", device="cuda")
    ev.compute_real_stats(ev.load_videos(real, data_type="video_numpy"))
    res = {}
    for d in args.fake_dirs:
        name = os.path.basename(os.path.normpath(d))
        fake = dump(name, [read(os.path.join(d, f"{tg}_gen.mp4")) for tg in common])
        ev.empty_fake_stats()
        ev.compute_fake_stats(ev.load_videos(fake, data_type="video_numpy"))
        res[name] = float(ev.compute_fvd_from_stats())
        print(f"[fvd] {name}: FVD {res[name]:.2f}", flush=True)
    json.dump({"n_pairs": len(common), "clip_len": 16, "clips_per_video": 5, "resolution": 224,
               "gt_from": args.gt_from, "fvd": res}, open(args.out, "w"), indent=1)
    print("[fvd] wrote", args.out)


if __name__ == "__main__":
    main()
