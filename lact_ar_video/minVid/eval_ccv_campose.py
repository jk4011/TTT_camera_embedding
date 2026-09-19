"""Stage 2 of the ccv camera-accuracy metric: RotErr / TransErr / CamMC.

Estimates the camera trajectory FROM each generated video with COLMAP and compares it to
the trajectory the model was conditioned on. Pixel metrics (PSNR/SSIM/LPIPS) say the frames
look like the target view; these say the rendered camera IS the requested camera.

Runs in the isolated sfm environment (numpy + pycolmap only):
  /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/sfm/bin/python eval_ccv_campose.py \
      --prep ../outputs/eval/campose_ccv_base_13999 --which gen --out campose.json

Protocol (stated here because the numbers are only comparable under it):
  * poses are taken at the frames the conditioning is defined on (0, 4, ..., 80) and made
    relative to frame 0, which removes COLMAP's arbitrary world gauge exactly;
  * monocular SfM has no scale, so one scalar is fitted by least squares on the
    translations (s = <t_pred, t_gt> / <t_pred, t_pred>) and the GT translations are
    normalised by max_j ||t_gt_j|| first, as in the CameraCtrl/CamI2V convention;
  * RotErr_j  = angle between R_gt_j and R_pred_j = arccos((tr(R_gt^T R_pred) - 1) / 2)
    TransErr_j = ||t_gt_j - s t_pred_j||
    CamMC_j    = ||[R_gt_j | t_gt_j] - [R_pred_j | s t_pred_j]||_F
    reported as the mean over j = 1..20 (j = 0 is the identity by construction);
  * a video SfM cannot register is a real failure mode of the generator, not a missing
    number: `registered_frac` is reported per pair, pairs below --min_reg are excluded
    from the means and counted in `n_failed`. Run `--which gt` for the floor: the same
    pipeline on the real target-camera video.
"""
import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import pycolmap
from PIL import Image

# MultiCamVideo renders at f/2.4-f/10 with a 50mm-equivalent lens, so most of the frame is
# out-of-focus bokeh: COLMAP's default SIFT finds ~300 keypoints on the harder clips and
# the mapper cannot even find an initial pair. Dropping the peak threshold to 0.001 takes
# the same frame to ~1800.
def _sift_options(peak=0.001, max_features=20000):
    o = pycolmap.FeatureExtractionOptions()
    o.sift.peak_threshold = peak
    o.sift.max_num_features = max_features
    o.sift.edge_threshold = 20.0
    return o


def twoview_chain_rotations(image_dir, K, frame_ids, device="cpu", peak=0.001):
    """Rotation of each conditioning frame w.r.t. the first, by chaining two-view geometry.

    For a camera that only pans, SfM has no baseline to triangulate and fails outright;
    23 of this dataset's 64 held-out target cameras are exactly that (pure rotation). Two-view
    geometry still recovers the rotation there (COLMAP falls back to the panoramic
    configuration), but only between frames that overlap: with a 27 degree field of view a
    33 degree pan leaves frame 0 and frame 80 sharing nothing, so this composes CONSECUTIVE
    conditioning frames instead. Measured on a real target-camera video, that recovers all
    20 frames with a mean error of 1.9 degrees (drifting to 3.8 by the end) -- the floor any
    generated number here is read against, which is why `--which gt` exists.
    """
    dev = pycolmap.Device.cpu if device == "cpu" else pycolmap.Device.cuda
    ext = pycolmap.FeatureExtractor.create(_sift_options(peak), device=dev)
    matcher = pycolmap.FeatureMatcher.create(device=dev)
    cam = pycolmap.Camera.create_from_model_id(1, 1, K[0][0], 832, 480)
    cam.params = [K[0][0], K[1][1], K[0][2], K[1][2]]
    cache = {}

    def feats(i):
        if i not in cache:
            path = os.path.join(image_dir, f"frame{i:04d}.jpg")
            if not os.path.isfile(path):
                cache[i] = None
            else:
                im = np.asarray(Image.open(path).convert("L"))
                k, d = ext.extract_from_uint8_array(im)
                cache[i] = (np.array([[p.x, p.y] for p in k], dtype=np.float64), k, d)
        return cache[i]

    out = {0: np.eye(3)}
    acc = np.eye(3)
    for j in range(1, len(frame_ids)):
        fa, fb = feats(frame_ids[j - 1]), feats(frame_ids[j])
        if fa is None or fb is None:
            break
        m = matcher.match(fa[1], fa[2], fb[1], fb[2])
        g = pycolmap.estimate_calibrated_two_view_geometry(cam, fa[0], cam, fb[0], m)
        if not pycolmap.estimate_two_view_geometry_pose(cam, fa[0], cam, fb[0], g):
            break
        if g.cam2_from_cam1 is None:
            break
        acc = g.cam2_from_cam1.rotation.matrix() @ acc
        out[j] = acc.copy()
    return out


def colmap_poses(image_dir, K, matcher="sequential", device="cpu", overlap=10):
    """{image_name: 4x4 c2w} from COLMAP, intrinsics fixed to K."""
    work = tempfile.mkdtemp(prefix="capose_")
    try:
        db = os.path.join(work, "db.db")
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        ro = pycolmap.ImageReaderOptions()
        ro.camera_model = "PINHOLE"
        ro.camera_params = f"{fx},{fy},{cx},{cy}"
        dev = pycolmap.Device.cpu if device == "cpu" else pycolmap.Device.cuda
        pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=ro, extraction_options=_sift_options(),
                                  device=dev)
        if matcher == "exhaustive":
            pycolmap.match_exhaustive(db, device=dev)
        else:
            po = pycolmap.SequentialPairingOptions()
            po.overlap = overlap
            po.quadratic_overlap = True
            po.loop_detection = False
            pycolmap.match_sequential(db, pairing_options=po, device=dev)
        opts = pycolmap.IncrementalPipelineOptions()
        opts.ba_refine_focal_length = False
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params = False
        recs = pycolmap.incremental_mapping(db, image_dir, os.path.join(work, "sparse"), options=opts)
        if not recs:
            return {}
        rec = max(recs.values(), key=lambda r: r.num_reg_images())
        out = {}
        for img in rec.images.values():
            w2c = np.eye(4)
            w2c[:3, :4] = img.cam_from_world().matrix()
            out[img.name] = np.linalg.inv(w2c)
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def relative_to_first(poses):
    """[N,4,4] -> poses expressed in the first camera's frame."""
    return np.linalg.inv(poses[0])[None] @ poses


def rot_metrics(R_gt, R_pred):
    """RotErr for a rotation-only camera; CamMC reduces to the rotation block."""
    tr = np.trace(np.transpose(R_gt, (0, 2, 1)) @ R_pred, axis1=1, axis2=2)
    ang = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    cammc = np.sqrt(((R_gt - R_pred) ** 2).sum(axis=(1, 2)))
    return {"RotErr_deg": float(np.degrees(ang).mean()), "RotErr_rad": float(ang.mean()),
            "TransErr": None, "CamMC": float(cammc.mean())}


def pose_metrics(c2w_gt, c2w_pred):
    g = relative_to_first(c2w_gt)
    p = relative_to_first(c2w_pred)
    tg, tp = g[1:, :3, 3], p[1:, :3, 3]
    norm = np.linalg.norm(tg, axis=-1).max()
    if norm > 1e-8:
        tg = tg / norm
    den = float((tp * tp).sum())
    s = float((tp * tg).sum() / den) if den > 1e-12 else 0.0
    tp = s * tp
    Rg, Rp = g[1:, :3, :3], p[1:, :3, :3]
    tr = np.trace(np.transpose(Rg, (0, 2, 1)) @ Rp, axis1=1, axis2=2)
    ang = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    trans = np.linalg.norm(tg - tp, axis=-1)
    dR = Rg - Rp
    cammc = np.sqrt((dR ** 2).sum(axis=(1, 2)) + ((tg - tp) ** 2).sum(axis=-1))
    return {"RotErr_deg": float(np.degrees(ang).mean()), "RotErr_rad": float(ang.mean()),
            "TransErr": float(trans.mean()), "CamMC": float(cammc.mean()), "scale": s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True, help="eval_ccv_campose_prep.py output dir")
    ap.add_argument("--which", default="gen", choices=["gen", "gt"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_pairs", type=int, default=0)
    ap.add_argument("--matcher", default="sequential", choices=["sequential", "exhaustive"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--min_reg", type=float, default=0.9,
                    help="min fraction of the 21 conditioning frames COLMAP must register")
    ap.add_argument("--static_thresh", type=float, default=1e-3,
                    help="GT translation extent below which the camera is treated as "
                         "rotation-only (SfM has no baseline there)")
    args = ap.parse_args()

    with open(os.path.join(args.prep, "pairs.json")) as f:
        meta = json.load(f)
    frame_ids = meta["pose_frame_ids"]
    pairs = meta["pairs"][:args.n_pairs] if args.n_pairs else meta["pairs"]
    out_path = args.out or os.path.join(args.prep, f"campose_{args.which}.json")

    per_pair, ok = [], []
    for pr in pairs:
        img_dir = os.path.join(args.prep, pr["tag"], args.which)
        if not os.path.isdir(img_dir):
            print(f"[campose] MISSING {img_dir}", flush=True)
            continue
        gt_all = np.asarray(pr["c2w_tgt"], dtype=np.float64)
        extent = float(np.linalg.norm(gt_all[:, :3, 3] - gt_all[0, :3, 3], axis=-1).max())
        rec = {"tag": pr["tag"], "index": pr["index"], "gt_translation_extent": extent}

        if extent < args.static_thresh:
            # rotation-only camera: no baseline, so SfM cannot run at all
            Rs = twoview_chain_rotations(img_dir, pr["K"], frame_ids, args.device)
            idx = sorted(Rs)
            rec["mode"] = "rotation_only"
            rec["registered_frac"] = len(idx) / len(frame_ids)
            if len(idx) >= args.min_reg * len(frame_ids):
                g = relative_to_first(gt_all[idx])
                rec.update(rot_metrics(np.transpose(g[1:, :3, :3], (0, 2, 1)),
                                       np.stack([Rs[j] for j in idx[1:]])))
                rec["n_frames"] = len(idx)
                ok.append(rec)
                print(f"[campose] {pr['tag']}: rotation-only, {len(idx)}/{len(frame_ids)} frames, "
                      f"RotErr {rec['RotErr_deg']:.2f} deg", flush=True)
            else:
                rec["failed"] = True
                print(f"[campose] {pr['tag']}: rotation-only chain broke at {len(idx)} frames", flush=True)
            per_pair.append(rec)
            with open(out_path, "w") as f:
                json.dump({"which": args.which, "n_pairs": len(per_pair), "per_pair": per_pair}, f, indent=1)
            continue

        poses = colmap_poses(img_dir, pr["K"], args.matcher, args.device)
        names = [f"frame{i:04d}.jpg" for i in frame_ids]
        got = [n in poses for n in names]
        frac = float(np.mean(got))
        rec["mode"] = "sfm"
        rec["registered_frac"] = frac
        if frac >= args.min_reg and got[0]:
            idx = [j for j, g in enumerate(got) if g]
            gt = np.asarray(pr["c2w_tgt"], dtype=np.float64)[idx]
            pred = np.stack([poses[names[j]] for j in idx])
            rec.update(pose_metrics(gt, pred))
            rec["n_frames"] = len(idx)
            ok.append(rec)
            print(f"[campose] {pr['tag']}: reg {frac:.2f}  RotErr {rec['RotErr_deg']:.2f} deg  "
                  f"TransErr {rec['TransErr']:.4f}  CamMC {rec['CamMC']:.4f}", flush=True)
        else:
            rec["failed"] = True
            print(f"[campose] {pr['tag']}: SfM FAILED (registered {frac:.2f})", flush=True)
        per_pair.append(rec)
        with open(out_path, "w") as f:
            json.dump({"which": args.which, "n_pairs": len(per_pair), "per_pair": per_pair}, f, indent=1)

    summary = {"which": args.which, "n_pairs": len(per_pair), "n_scored": len(ok),
               "n_failed": len(per_pair) - len(ok),
               "registered_frac_mean": float(np.mean([r["registered_frac"] for r in per_pair]))
               if per_pair else 0.0}
    sfm_ok = [r for r in ok if r.get("mode") == "sfm"]
    summary["n_rotation_only"] = len([r for r in ok if r.get("mode") == "rotation_only"])
    summary["n_sfm"] = len(sfm_ok)
    for k in ("RotErr_deg", "RotErr_rad", "CamMC"):
        summary[k] = float(np.mean([r[k] for r in ok])) if ok else None
    # translation is undefined for a rotation-only camera, so it averages over the SfM subset
    summary["TransErr"] = float(np.mean([r["TransErr"] for r in sfm_ok])) if sfm_ok else None
    with open(out_path, "w") as f:
        json.dump({**summary, "per_pair": per_pair}, f, indent=1)
    print("[campose] summary:", json.dumps({k: v for k, v in summary.items()}), flush=True)


if __name__ == "__main__":
    main()
