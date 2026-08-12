#!/bin/bash
# Q42 goal watcher: exits when gobj_imgvo lands (or after 3.5 h), printing the verdict
# against the fair bar. Harness-tracked so Claude wakes for it.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for i in $(seq 1 84); do [ -f outputs/gobj_imgvo_s95/eval.json ] && break; sleep 150; done
/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python - <<'PY'
import json, math, statistics as st
try:
    d=json.load(open("outputs/gobj_imgvo_s95/eval.json"))
except Exception:
    print("imgvo still not done after 3.5 h -- check the trainer"); raise SystemExit
praw=[json.load(open(f"outputs/gobj_prope_raw_{s}/eval.json"))["psnr"] for s in ["s95","s137","s211"]]
bar=st.mean(praw)
print(f"=== Q42 VERDICT ===")
print(f"imgvo (ours: img-rope + rot transport): {d['psnr']:.3f} / {d['lpips']:.4f}")
print(f"fair bar (prope_raw 3-seed):            {bar:.3f} +- {st.stdev(praw):.3f}")
print(f"delta vs bar: {d['psnr']-bar:+.3f}  ->  {'CLEARS THE BAR' if d['psnr']>bar else 'below the bar'}")
a=d["per_scene_psnr"]; b=json.load(open("outputs/gobj_prope_raw_s95/eval.json"))["per_scene_psnr"]
if len(a)==len(b):
    dd=[x-y for x,y in zip(a,b)]; m=sum(dd)/len(dd)
    sd=math.sqrt(sum((x-m)**2 for x in dd)/(len(dd)-1))
    print(f"paired vs prope_raw s95: {m:+.3f} (t={m/(sd/math.sqrt(len(dd))):+.2f})")
PY
