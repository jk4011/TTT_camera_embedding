#!/bin/bash
# Q38 goal check: wait for BOTH ogta cells' eval.json, then print the verdict table
# against each dataset's incumbents. Exits when both are in (harness notifies).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
until [ -f outputs/gobj_ogta_s95/eval.json ] && [ -f outputs/ogta_s95/eval.json ]; do sleep 180; done
/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python - <<'PY'
import json
def g(p):
    try: d=json.load(open(p)); return d["psnr"], d["lpips"]
    except Exception: return None, None
print("=== Q38 verdict ===")
print("gObjaverse (91 deg -- every sinusoidal arm lost here, F51):")
for n,p in [("base","outputs/gobj_base_s95/eval.json"),
            ("prope_orig (affine GA)","outputs/gobj_prope_orig_s95/eval.json"),
            ("gta_in (affine GA)","outputs/gobj_gta_in_s95/eval.json"),
            ("OGTA (orthogonal GA)","outputs/gobj_ogta_s95/eval.json")]:
    ps,lp=g(p); print(f"  {n:24s} {ps if ps else '--':>8} {lp if lp else '':>8}")
print("RE10K (7 deg -- the ladder wins here: input +0.60, both +1.07):")
for n,p in [("base","outputs/base_s95/eval.json"),
            ("input ladder (pra_hi)","outputs/pra_hi_s95/eval.json"),
            ("OGTA (orthogonal GA)","outputs/ogta_s95/eval.json")]:
    ps,lp=g(p); print(f"  {n:24s} {ps if ps else '--':>8} {lp if lp else '':>8}")
PY
