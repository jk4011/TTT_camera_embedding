"""Paired per-scene comparison of eval.json files (house standard: seed-matched, paired t).

  python paired_eval.py <ref_eval.json> <cell_eval.json> [<cell_eval.json> ...] [--md]

Prints, per cell vs the reference: mean PSNR/SSIM/LPIPS, paired delta, paired t, win%.
Scene lists must have identical length and order (same test index, same eval protocol);
a length mismatch aborts rather than silently misaligning.
"""
import json
import sys

import numpy as np


def load(p):
    j = json.load(open(p))
    return {k: np.asarray(j[f"per_scene_{k}"], dtype=np.float64)
            for k in ("psnr", "lpips", "ssim") if f"per_scene_{k}" in j}, j


def paired(a, b):
    d = b - a
    n = len(d)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n) + 1e-12)
    return d.mean(), t, (d > 0).mean() * 100.0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    md = "--md" in sys.argv
    ref_arr, ref_j = load(args[0])
    n = len(ref_arr["psnr"])
    print(f"reference: {args[0]}  n={n}  PSNR {ref_arr['psnr'].mean():.3f}  "
          f"LPIPS {ref_arr['lpips'].mean():.4f}"
          + (f"  SSIM {ref_arr['ssim'].mean():.4f}" if "ssim" in ref_arr else ""))
    if md:
        print("| cell | PSNR | dPSNR (t, win%) | LPIPS | dLPIPS (t) | SSIM | dSSIM (t) |")
        print("|---|---|---|---|---|---|---|")
    for p in args[1:]:
        arr, _ = load(p)
        if len(arr["psnr"]) != n:
            print(f"!! {p}: n={len(arr['psnr'])} != {n} (misaligned scene sets) -- skipped")
            continue
        dp, tp, wp = paired(ref_arr["psnr"], arr["psnr"])
        dl, tl, wl = paired(ref_arr["lpips"], arr["lpips"])
        row = f"{p.split('/')[-2] if '/' in p else p}"
        if md:
            s = f"| {row} | {arr['psnr'].mean():.3f} | {dp:+.3f} ({tp:+.1f}, {wp:.0f}%) | " \
                f"{arr['lpips'].mean():.4f} | {dl:+.4f} ({tl:+.1f}) |"
            if "ssim" in arr and "ssim" in ref_arr:
                ds, ts, _ = paired(ref_arr["ssim"], arr["ssim"])
                s += f" {arr['ssim'].mean():.4f} | {ds:+.4f} ({ts:+.1f}) |"
            else:
                s += " - | - |"
            print(s)
        else:
            print(f"{row:32s} PSNR {arr['psnr'].mean():.3f} d={dp:+.3f} t={tp:+.2f} win={wp:.1f}%  "
                  f"LPIPS {arr['lpips'].mean():.4f} d={dl:+.4f} t={tl:+.2f}")


if __name__ == "__main__":
    main()
