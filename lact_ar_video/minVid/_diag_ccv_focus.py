"""CCV: how often does the scene focus p* (from the SRC trajectory) leave rays with t_c <= 0.02?"""
import json, sys, torch
sys.path.insert(0, "..")
from eval_ccv_common import load_config, load_or_build_pairs, make_pair_dataset, DEFAULT_PAIRS_JSON
from minVid.models.blocks.cam_phase_builder import build_ccv_capet_inputs
cfg = load_config("configs/ar/abl_ccv_capet.yaml")
pairs = load_or_build_pairs(DEFAULT_PAIRS_JSON, cfg)["pairs"]
ds = make_pair_dataset(cfg, pairs)
ds._decode_clip = lambda path: (None, (832/1280, 480/720, 10.5, 0, 1280, 720))   # poses only, skip video
HW, n_src = 30 * 52, 21
rows = []
for i in range(len(pairs)):
    vdir, rel, sc, tc_ = ds.pairs[i]
    ex = ds._load_extrinsics(rel)
    c_s, c_t = ds._poses_for_cam(ex, sc), ds._poses_for_cam(ex, tc_)
    from minVid.data.multicam_pair_dataset import normalize_with_mean_pose
    allc = normalize_with_mean_pose(torch.cat([c_s, c_t], 0)); c_s, c_t = allc[:21], allc[21:]
    K = ds._intrinsics(rel, (832/1280, 480/720, 10.5, 0, 1280, 720))
    c7 = build_ccv_capet_inputs(c_s, c_t, K, latent_hw=(30, 52), n_latent_f=21, ar_window_f=3)
    t = c7[:, 6]
    src_rot = float((c_s[:, :3, 3] - c_s[0, :3, 3]).norm(dim=-1).max()) < 1e-3
    f = c_s[:, :3, 2]; f = f / f.norm(dim=-1, keepdim=True)
    pan = float(torch.rad2deg(torch.arccos((f[0] * f).sum(-1).clamp(-1, 1))).max())
    rows.append((src_rot, float((t[:n_src*HW] <= 0.02).float().mean()), float((t[n_src*HW:] <= 0.02).float().mean()), pan))
for name, sel in (("src camera pure rotation", True), ("src camera translates", False)):
    r = [x for x in rows if x[0] == sel]
    if not r: continue
    med = lambda v: sorted(v)[len(v)//2]
    print(f"{name:26s} n={len(r):2d}  clamped SRC tokens: median {med([x[1] for x in r]):.0%} max {max(x[1] for x in r):.0%}"
          f"  | clamped TGT tokens: median {med([x[2] for x in r]):.0%} max {max(x[2] for x in r):.0%}")

print("\npure-rotation source pairs, sorted by pan angle:")
for r in sorted([x for x in rows if x[0]], key=lambda x: x[3]):
    print(f"  pan {r[3]:5.1f} deg   clamped SRC {r[1]:4.0%}   TGT {r[2]:4.0%}")
bad = [x for x in rows if x[1] > 0.5 or x[2] > 0.5]
print(f"\npairs with >50% of SRC or TGT tokens collapsed: {len(bad)} / {len(rows)}")
