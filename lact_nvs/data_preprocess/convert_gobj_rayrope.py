"""Convert our gObjaverse WAI renders into RayRoPE's Objaverse layout.

Target layout (nvs/objaverse_dataset.py):
  <OBJV_DIR>/<uid>/cameras.json        {"cameras": [{"view_id", "image_name",
      "intrinsics": {"focal_length": [fx, fy], "principal_point": [cx, cy]},
      "extrinsics": {"camera_to_world": 3x4}}, ...]}
  <OBJV_DIR>/<uid>/views/<image_name>  RGB (white-composited), pre-resized 256

Their parser post-multiplies camera_to_world by blender2opencv = diag(1,-1,-1,1),
so we store c2w_opencv @ blender2opencv (the matrix is its own inverse) to make
their parse yield exactly our opencv c2w. Images are pre-resized to 256x256 with
intrinsics scaled accordingly, so their resize_crop is an identity.

Index files (same schema as their assets/objaverse_index_*):
  train: context = 2 seeded-random views per object, targets = the rest
  test:  context = frames [7, 32], targets = [15, 20, 27]  (our prope-testbed
         protocol, for cross-testbed comparability)
Split: same as convert_gobj_prope.py -- last 500 sorted uids are the test set.
"""
import json
import os
import random
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

SRC = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/gobjaverse_wai"
DST = "/tmp/rayrope_objv"
SIZE = 256
B2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def convert_scene(sdir):
    uid = os.path.basename(sdir.rstrip("/"))
    meta = json.load(open(os.path.join(sdir, "scene_meta.json")))
    odir = os.path.join(DST, uid)
    vdir = os.path.join(odir, "views")
    os.makedirs(vdir, exist_ok=True)
    cams = []
    for i, fr in enumerate(meta["frames"]):
        name = f"{i:03d}.jpg"
        opath = os.path.join(vdir, name)
        if not os.path.exists(opath):
            img = Image.open(os.path.join(sdir, fr["file_path"]))
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            else:
                img = img.convert("RGB")
            w0, h0 = img.size
            img = img.resize((SIZE, SIZE), Image.LANCZOS)
            img.save(opath, quality=92)
        else:
            w0, h0 = fr["w"], fr["h"]
        sx, sy = SIZE / fr["w"], SIZE / fr["h"]
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        c2w_store = (c2w @ B2CV)[:3, :4]
        cams.append({
            "view_id": i,
            "image_name": name,
            "intrinsics": {
                "focal_length": [fr["fl_x"] * sx, fr["fl_y"] * sy],
                "principal_point": [fr["cx"] * sx, fr["cy"] * sy],
            },
            "extrinsics": {"camera_to_world": c2w_store.tolist()},
        })
    json.dump({"cameras": cams}, open(os.path.join(odir, "cameras.json"), "w"))
    return uid, len(cams)


def main():
    scenes = sorted(
        os.path.join(SRC, d) for d in os.listdir(SRC)
        if os.path.isfile(os.path.join(SRC, d, "scene_meta.json"))
    )
    print(f"{len(scenes)} scenes")
    test_uids = set(os.path.basename(s) for s in scenes[-500:])
    os.makedirs(DST, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    results = []
    with Pool(64) as pool:
        for i, r in enumerate(pool.imap_unordered(convert_scene, scenes, chunksize=8)):
            results.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(scenes)}", flush=True)

    rng = random.Random(95)
    train_idx, test_idx = {}, {}
    for uid, n in results:
        names = [f"{i:03d}.jpg" for i in range(n)]
        if uid in test_uids:
            test_idx[uid] = {
                "context_view_files": [names[7], names[32]],
                "target_view_files": [names[15], names[20], names[27]],
            }
        else:
            ctx = rng.sample(range(n), 2)
            train_idx[uid] = {
                "context_view_files": [names[c] for c in ctx],
                "target_view_files": [names[i] for i in range(n) if i not in ctx],
            }
    json.dump(train_idx, open(os.path.join(DST, "index_train_context2.json"), "w"))
    json.dump(test_idx, open(os.path.join(DST, "index_test_context2.json"), "w"))
    print(f"DONE: train {len(train_idx)}, test {len(test_idx)} -> {DST}")


if __name__ == "__main__":
    main()
