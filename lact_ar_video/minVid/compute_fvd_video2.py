"""FVD for the video2 4-arm generation eval (cd-fvd, classic I3D features).

Real side: a folder of held-out real mp4s (stats cached to --real_stats so the
arms share one real pass). Fake side: one or more folders of generated mp4s
(one per arm). Same loader settings both sides (224 px, 16-frame windows).

Run: python compute_fvd_video2.py --real_dir <dir> --real_stats <pkl> \
       --fake_dirs <dir1> <dir2> ... --out <json>
"""
import argparse
import json
import os

# cd-fvd expects the old scipy sqrtm(..., disp=False) tuple API
import scipy.linalg as sla
_orig_sqrtm = sla.sqrtm
def _sqrtm(A, disp=None, **kw):
    r = _orig_sqrtm(A)
    return (r, 0.0) if disp is not None else r
sla.sqrtm = _sqrtm

from cdfvd import fvd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", type=str, required=True)
    ap.add_argument("--real_stats", type=str, required=True,
                    help="pkl cache for the real-side statistics")
    ap.add_argument("--fake_dirs", type=str, nargs="+", required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    ev = fvd.cdfvd("i3d", n_real="full", n_fake="full", device=args.device)
    if os.path.isfile(args.real_stats):
        ev.load_real_stats(args.real_stats)
        print(f"[fvd] real stats loaded from {args.real_stats}")
    else:
        ev.compute_real_stats(ev.load_videos(
            args.real_dir, data_type="video_folder",
            resolution=args.resolution, sequence_length=args.seq_len))
        ev.save_real_stats(args.real_stats)
        print(f"[fvd] real stats computed from {args.real_dir}")

    results = {}
    for fd in args.fake_dirs:
        ev.empty_fake_stats()
        ev.compute_fake_stats(ev.load_videos(
            fd, data_type="video_folder",
            resolution=args.resolution, sequence_length=args.seq_len))
        v = float(ev.compute_fvd_from_stats())
        results[os.path.basename(os.path.normpath(fd))] = v
        print(f"[fvd] {fd}: {v:.2f}", flush=True)

    json.dump({"resolution": args.resolution, "seq_len": args.seq_len,
               "real_dir": args.real_dir, "fvd": results},
              open(args.out, "w"), indent=1)
    print(f"DONE -> {args.out}")


if __name__ == "__main__":
    main()
