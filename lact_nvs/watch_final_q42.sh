#!/bin/bash
# FINAL Q42 verdict: waits for imgvo s137+s211, prints 3-seed mean-vs-mean vs the bar.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
until [ -f outputs/gobj_imgvo_s137/eval.json ] && [ -f outputs/gobj_imgvo_s211/eval.json ]; do sleep 240; done
/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python - <<'PY'
import json, statistics as st
iv=[json.load(open(f"outputs/gobj_imgvo_{s}/eval.json"))["psnr"] for s in ["s95","s137","s211"]]
pr=[json.load(open(f"outputs/gobj_prope_raw_{s}/eval.json"))["psnr"] for s in ["s95","s137","s211"]]
print("=== Q42 FINAL: 3-seed mean vs mean ===")
print(f"imgvo    : {st.mean(iv):.3f} +- {st.stdev(iv):.3f}  {[round(x,3) for x in iv]}")
print(f"prope_raw: {st.mean(pr):.3f} +- {st.stdev(pr):.3f}  {[round(x,3) for x in pr]}")
d=st.mean(iv)-st.mean(pr)
print(f"delta = {d:+.3f}  ->  {'GOAL MET (mean over mean)' if d>0 else 'goal not met'}")
PY
