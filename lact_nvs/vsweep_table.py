"""Compile the input-view sweep: per dataset, per view count V, paired delta of each arm vs base.
Usage: python vsweep_table.py [--md]   (reads outputs/<exp>/eval_<ds>_nv<V>.json written by run_vsweep.sh)"""
import json, os, sys, numpy as np
DS = {
  "re10k":    dict(base="base_s137", input="pra_hi_s137", hidden="h_pra_hi_s137", both="pra_h_hi_s137"),
  "dl3dvw48": dict(base="dl3dvw48_base_s137", input="dl3dvw48_input_s137", hidden="dl3dvw48_hidden_s137", both="dl3dvw48_both_s137"),
  "gobjv60":  dict(base="gobjvi_base_s95", input="gobjvi_input_s95", hidden="gobjvi_hidden_s95", both="gobjvi_both_s95"),
}
VIEWS = [4, 8, 12, 20, 32, 48]
def load(ds, exp, v):
    p = f"outputs/{exp}/eval_{ds}_nv{v}.json"
    if not os.path.isfile(p): return None
    j = json.load(open(p)); return np.array(j["per_scene_psnr"]), np.array(j["per_scene_lpips"])
for ds, arms in DS.items():
    print(f"\n### {ds}  (PSNR; delta vs base, paired t; LPIPS delta)")
    print("| V | base | input | hidden | both (TTT-RoPE) |"); print("|---|---|---|---|---|")
    for v in VIEWS:
        b = load(ds, arms["base"], v)
        if b is None: print(f"| {v} | -- | | | |"); continue
        row = [f"{b[0].mean():.3f}"]
        for a in ("input", "hidden", "both"):
            r = load(ds, arms[a], v)
            if r is None or len(r[0]) != len(b[0]): row.append("--"); continue
            d = r[0] - b[0]; t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-9)
            dl = (r[1] - b[1]).mean()
            row.append(f"{r[0].mean():.3f} ({d.mean():+.3f}, t={t:+.1f}; L {dl:+.4f})")
        print(f"| {v} | " + " | ".join(row) + " |")
