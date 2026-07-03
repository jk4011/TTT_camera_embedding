"""Reshard pixelsplat-style RE10K .torch chunks into per-scene files on /tmp (tmpfs).

Each source chunk is a list of ~16 scenes:
  {url, timestamps [N], cameras [N, 18], images: list of N jpeg-byte uint8 tensors, key}
cameras row: [fx fy cx cy 0 0, w2c(3x4) flattened]  (intrinsics normalized by image size)

Output: <odir>/<split>/<key>.torch  (single scene dict, same fields)
        <odir>/<split>_index.json   list of {"file": ..., "num_frames": N}
"""
import argparse
import json
import os
from multiprocessing import Pool

import torch


def process_chunk(job):
    chunk_path, out_dir = job
    entries = []
    try:
        scenes = torch.load(chunk_path, weights_only=False, map_location="cpu")
    except Exception as e:
        print(f"FAILED to load {chunk_path}: {e}")
        return []
    for scene in scenes:
        key = scene["key"]
        out_path = os.path.join(out_dir, f"{key}.torch")
        if not os.path.exists(out_path):
            torch.save(scene, out_path)
        entries.append({"file": f"{key}.torch", "num_frames": len(scene["images"])})
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="dir with *.torch chunks")
    parser.add_argument("--odir", type=str, required=True, help="output dir for per-scene files")
    parser.add_argument("--index", type=str, required=True, help="output index json path")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)
    chunks = sorted(
        os.path.join(args.src, f) for f in os.listdir(args.src) if f.endswith(".torch")
    )
    print(f"{len(chunks)} chunks -> {args.odir}")

    all_entries = []
    with Pool(args.workers) as pool:
        for i, entries in enumerate(
            pool.imap_unordered(process_chunk, [(c, args.odir) for c in chunks])
        ):
            all_entries.extend(entries)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(chunks)} chunks, {len(all_entries)} scenes", flush=True)

    all_entries.sort(key=lambda e: e["file"])
    with open(args.index, "w") as f:
        json.dump(all_entries, f)
    print(f"DONE: {len(all_entries)} scenes, index -> {args.index}")


if __name__ == "__main__":
    main()
