#!/bin/bash
# Final goal watcher: exits when prope75, prope_in and the two bar seeds are ALL in,
# printing the complete goal table. Harness-tracked so Claude wakes for the verdict.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
until [ -f outputs/gobj_prope75_s95/eval.json ] \
   && [ -f outputs/gobj_prope_in_s95/eval.json ] \
   && [ -f outputs/gobj_prope_raw_s137/eval.json ] \
   && [ -f outputs/gobj_prope_raw_s211/eval.json ]; do sleep 300; done
/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python - <<'PY'
import json, statistics as st
def g(e):
    try: return json.load(open(f"outputs/{e}/eval.json"))["psnr"]
    except Exception: return None
print("=== FINAL GOAL TABLE (gObjaverse) ===")
for n, es in [("base", ["gobj_base_s95","gobj_base_s137","gobj_base_s211"]),
              ("prope_raw (fair bar)", ["gobj_prope_raw_s95","gobj_prope_raw_s137","gobj_prope_raw_s211"]),
              ("prope_orig", ["gobj_prope_orig_s95","gobj_prope_orig_s137","gobj_prope_orig_s211"]),
              ("imgrope", ["gobj_prope_imgrope_s95"]),
              ("prope75 (ours-tuned)", ["gobj_prope75_s95"]),
              ("prope_in (no v/o)", ["gobj_prope_in_s95"]),
              ("h-GA", ["gobj_hga_s95"])]:
    vs=[g(e) for e in es]; vs=[v for v in vs if v]
    if vs:
        m=sum(vs)/len(vs)
        sd=f" +- {st.stdev(vs):.3f}" if len(vs)>2 else ""
        print(f"  {n:24s} {m:.3f}{sd}  (n={len(vs)})")
PY
