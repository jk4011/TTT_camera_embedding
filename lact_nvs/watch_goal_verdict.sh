#!/bin/bash
# Goal watcher (beat prope on gObjaverse): exits when the h-GA gobj cell and the
# s211 base land, printing the full goal table. Harness-tracked, so Claude wakes.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
until [ -f outputs/gobj_hga_s95/eval.json ] && [ -f outputs/gobj_base_s211/eval.json ]; do sleep 240; done
/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python - <<'PY'
import json
def g(e):
    try: return json.load(open(f"outputs/{e}/eval.json"))["psnr"]
    except Exception: return None
print("=== GOAL TABLE: beat prope on gObjaverse ===")
rows=[("base s95/137/211", ["gobj_base_s95","gobj_base_s137","gobj_base_s211"]),
      ("prope_orig 3-seed", ["gobj_prope_orig_s95","gobj_prope_orig_s137","gobj_prope_orig_s211"]),
      ("prope_raw (bar, 1 seed)", ["gobj_prope_raw_s95"]),
      ("imgrope", ["gobj_prope_imgrope_s95"]),
      ("h-GA (ours)", ["gobj_hga_s95"]),
      ("prope75", ["gobj_prope75_s95"]),
      ("prope_in", ["gobj_prope_in_s95"])]
for n, es in rows:
    vs=[g(e) for e in es]; vs=[v for v in vs if v]
    if vs: print(f"  {n:26s} {sum(vs)/len(vs):.3f}" + (f"  (n={len(vs)} seeds)" if len(es)>1 else ""))
    else: print(f"  {n:26s} pending")
PY
