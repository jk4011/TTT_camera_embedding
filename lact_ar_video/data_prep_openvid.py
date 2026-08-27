"""Prep the OpenVid-1M subset: unzip part100, match captions, probe + filter,
write the index json the OpenVidDataset loader reads.

Filters: landscape (w > h), height >= 480, native duration >= 5.3 s and
fps >= 14 (so 81 frames at ~16 fps fit without frame repetition), decodable.

Run: .venv_llm/bin/python data_prep_openvid.py
"""
import csv
import json
import os
import subprocess
import sys
import zipfile
from multiprocessing import Pool

ROOT = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/OpenVid-1M"
ZIPS = [os.path.join(ROOT, z) for z in ["OpenVid_part100.zip", "OpenVid_part99.zip", "OpenVid_part101.zip"]]
CSV = os.path.join(ROOT, "data", "train", "OpenVid-1M.csv")
VID = os.path.join(ROOT, "videos")
OUT = os.path.join(ROOT, "openvid_index.json")


def probe(name):
    import decord
    path = os.path.join(VID, name)
    try:
        vr = decord.VideoReader(path, num_threads=1)
        n = len(vr)
        fps = float(vr.get_avg_fps())
        h, w, _ = vr[0].shape
        del vr
    except Exception:
        return None
    if w <= h or h < 480 or fps < 14 or n / max(fps, 1e-6) < 5.3:
        return None
    return name


def main():
    os.makedirs(VID, exist_ok=True)
    for zpath in ZIPS:
        if not os.path.isfile(zpath):
            print(f"skip missing {zpath}", flush=True)
            continue
        zf = zipfile.ZipFile(zpath)
        names = [n for n in zf.namelist() if n.lower().endswith(".mp4")]
        todo = [n for n in names if not os.path.exists(
            os.path.join(VID, os.path.basename(n)))]
        print(f"{os.path.basename(zpath)}: {len(names)} mp4s, extracting {len(todo)}",
              flush=True)
        for k, n in enumerate(todo):
            with zf.open(n) as fsrc, open(
                    os.path.join(VID, os.path.basename(n)), "wb") as dst:
                while True:
                    buf = fsrc.read(1 << 22)
                    if not buf:
                        break
                    dst.write(buf)
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(todo)}", flush=True)
        zf.close()

    present = set(os.listdir(VID))
    print(f"{len(present)} extracted; matching captions ...", flush=True)
    captions = {}
    with open(CSV, newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            v = row["video"]
            if v in present:
                captions[v] = row["caption"]
    print(f"{len(captions)} matched captions; probing ...", flush=True)

    with Pool(48) as pool:
        kept = [r for r in pool.imap_unordered(probe, sorted(captions), chunksize=16)
                if r is not None]
    kept.sort()
    entries = [{"file": k, "caption": captions[k]} for k in kept]
    json.dump(entries, open(OUT, "w"))
    print(f"INDEX_DONE: {len(entries)} clips -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
