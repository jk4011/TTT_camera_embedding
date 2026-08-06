"""gobjaverse_wai scene_meta.json -> the PRoPE repo's transforms.json format.

Writes a parallel tree of tiny JSONs plus an `images` symlink per scene; no pixel
data is copied and the source dataset is never written to.

Their loader does c2w = transform_matrix @ blender2opencv (dataset.py:257) with
blender2opencv = diag(1,-1,-1,1). gobjaverse_wai stores OPENCV c2w, so we write
transform_matrix = c2w_opencv @ blender2opencv; the two cancel (B @ B = I) and their
pipeline sees exactly our verified opencv poses. Intrinsics are TOP-LEVEL in their
schema (shared per scene), which gobjaverse satisfies (shared_intrinsics: true).
"""
import json, os, sys
import numpy as np

SRC = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/gobjaverse_wai"
DST = "/tmp/gobj_prope"
B = np.diag([1.0, -1.0, -1.0, 1.0])

ks = sorted(k for k in os.listdir(SRC)
            if os.path.isdir(os.path.join(SRC, k))
            and os.path.isfile(os.path.join(SRC, k, "scene_meta.json")))
splits = {"train": ks[:-500], "test": ks[-500:]}   # same holdout as F51/Q36
for split, keys in splits.items():
    for i, k in enumerate(keys):
        sdir = os.path.join(DST, split, k)
        os.makedirs(sdir, exist_ok=True)
        img_link = os.path.join(sdir, "images")
        if not os.path.islink(img_link):
            os.symlink(os.path.join(SRC, k, "images"), img_link)
        out = os.path.join(sdir, "transforms.json")
        if os.path.exists(out):
            continue
        m = json.load(open(os.path.join(SRC, k, "scene_meta.json")))
        fr0 = m["frames"][0]
        meta = {
            "fl_x": fr0["fl_x"], "fl_y": fr0["fl_y"],
            "cx": fr0["cx"], "cy": fr0["cy"],
            "w": fr0["w"], "h": fr0["h"],
            "frames": [],
        }
        for fr in m["frames"]:
            c2w = np.array(fr["transform_matrix"], dtype=np.float64)
            meta["frames"].append({
                "file_path": os.path.join("images", os.path.basename(fr["image"])),
                "transform_matrix": (c2w @ B).tolist(),
            })
        json.dump(meta, open(out, "w"))
        if (i + 1) % 2000 == 0:
            print(f"  {split}: {i+1}/{len(keys)}", flush=True)
    print(f"{split}: {len(keys)} scenes done")

# evaluation index: deterministic 2 context + 3 target views for every test scene.
# Context pair [7, 32] has frame distance 25 = their selector's MINIMUM, i.e. the
# most favourable in-distribution geometry their own training regime produces.
idx = {k: {"context": [7, 32], "target": [15, 20, 27]} for k in splits["test"]}
os.makedirs("assets", exist_ok=True)
json.dump(idx, open("assets/evaluation_index_gobj.json", "w"))
print("eval index written: context [7,32], targets [15,20,27]")
