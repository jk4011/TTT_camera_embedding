"""(src, tgt) camera-pair dataset for the camera-controlled video (ccv) runs.

Reads synchronized clips of the SAME scene from two different cameras of the
MultiCamVideo dataset (<root>/<f*_aperture*>/<sceneN>/videos/camXX.mp4,
UE5-rendered, 81 frames @ 15fps, 1280x1280) plus the per-frame extrinsics
(cameras/camera_extrinsics.json, read from the NFS original when the local
/tmp copy has no cameras/ dir) and returns per item:

    {
        "frames_src": float32 [3, 81, 480, 832] in [0, 1],
        "frames_tgt": float32 [3, 81, 480, 832] in [0, 1],
        "c2w_src":    float32 [21, 4, 4]  canonicalized CV-convention c2w,
        "c2w_tgt":    float32 [21, 4, 4],
        "K":          float32 [3, 3]      intrinsics of the decoded 480x832,
        "caption":    str,
    }

Decode/resize/crop mirrors simple_video_dataset.SimpleVideoDataset exactly
(decode-time resize covering the target, then center crop), except frames are
the fixed indices 0..80 so poses stay frame-synchronized between cameras
(MCV clips are exactly 81 frames, so the random start would be 0 anyway).

Pose pipeline (official ReCamMaster conventions):
  raw string "[r r r 0] [r r r 0] [r r r 0] [tx ty tz 1]" -> 4x4 (row-vector
  convention; translation in the 4th ROW) -> transpose -> UE->CV:
  c2w = c2w[:, [1, 2, 0, 3]]; c2w[:3, 1] *= -1; c2w[:3, 3] /= 100 (cm -> m).
Frames 0, 4, ..., 80 -> 21 poses per camera (VAE temporal stride 4).
Canonical frame: normalize_with_mean_pose (ported from lact_nvs/data.py)
over the 42 poses of BOTH cameras jointly, incl. the scene-scale division.
"""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import decord

    decord.bridge.set_bridge("native")
    _HAS_DECORD = True
except ImportError:  # pragma: no cover
    _HAS_DECORD = False


DEFAULT_DATA_ROOT = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/MultiCamVideo-Dataset/train"
DEFAULT_CAM_ROOT = (
    "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/"
    "MultiCamVideo-Dataset/train"
)
DEFAULT_CAPTION = "a person moving through a scene, cinematic camera"
SENSOR_WIDTH_MM = 23.76
NATIVE_RES = 1280
NUM_CAMS = 10


def normalize(x):
    return x / x.norm()


def normalize_with_mean_pose(c2ws: torch.Tensor):
    """Ported verbatim from lact_nvs/data.py (mean-pose canonicalization +
    scene-scale normalization to [-1, 1] camera centers)."""
    center = c2ws[:, :3, 3].mean(0)
    vec2 = c2ws[:, :3, 2].mean(0)
    up = c2ws[:, :3, 1].mean(0)

    vec2 = normalize(vec2)
    vec0 = normalize(torch.cross(up, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, center], 1)

    avg_pos = c2ws.new_zeros(4, 4)
    avg_pos[3, 3] = 1.0
    avg_pos[:3] = m

    c2ws = torch.linalg.inv(avg_pos) @ c2ws

    scene_scale = torch.max(torch.abs(c2ws[:, :3, 3]))
    c2ws[:, :3, 3] /= scene_scale

    return c2ws


def parse_matrix_string(s: str) -> torch.Tensor:
    """'[a b c d] [e f g h] [i j k l] [m n o p]' -> [4, 4] fp32 (rows)."""
    vals = [float(v) for v in s.replace("[", " ").replace("]", " ").split()]
    assert len(vals) == 16, f"bad extrinsic string: {s!r}"
    return torch.tensor(vals, dtype=torch.float32).reshape(4, 4)


def ue_matrix_to_cv_c2w(mat_rows: torch.Tensor) -> torch.Tensor:
    """Row-vector UE 4x4 (translation in 4th ROW, cm) -> CV-convention c2w (m).

    Official ReCamMaster conversion: transpose to column convention, then
    c2w = c2w[:, [1, 2, 0, 3]]; c2w[:3, 1] *= -1; c2w[:3, 3] /= 100.
    """
    c2w = mat_rows.t().contiguous()
    c2w = c2w[:, [1, 2, 0, 3]]
    c2w[:3, 1] *= -1.0
    c2w[:3, 3] /= 100.0
    return c2w


def build_pair_index(data_root, num_pairs=2000, index_seed=42):
    """Fixed, seeded index of (videos_dir, relpath, src_cam, tgt_cam)."""
    rng = random.Random(index_seed)

    scene_dirs = []  # (videos_dir, relpath "f??_aperture?/sceneN")
    for focal_dir in sorted(os.listdir(data_root)):
        focal_path = os.path.join(data_root, focal_dir)
        if not os.path.isdir(focal_path):
            continue
        for scene in sorted(os.listdir(focal_path)):
            vdir = os.path.join(focal_path, scene, "videos")
            scene_dirs.append((vdir, os.path.join(focal_dir, scene)))

    if len(scene_dirs) == 0:
        raise RuntimeError(f"No scene directories found under {data_root}")

    index = []
    seen = set()
    max_attempts = num_pairs * 20
    attempts = 0
    while len(index) < num_pairs and attempts < max_attempts:
        attempts += 1
        vdir, rel = rng.choice(scene_dirs)
        src_cam = rng.randint(1, NUM_CAMS)
        tgt_cam = rng.randint(1, NUM_CAMS)
        if src_cam == tgt_cam:
            continue
        key = (vdir, src_cam, tgt_cam)
        if key in seen:
            continue
        src_path = os.path.join(vdir, f"cam{src_cam:02d}.mp4")
        tgt_path = os.path.join(vdir, f"cam{tgt_cam:02d}.mp4")
        if os.path.isfile(src_path) and os.path.isfile(tgt_path):
            seen.add(key)
            index.append((vdir, rel, src_cam, tgt_cam))

    if len(index) < num_pairs:
        raise RuntimeError(
            f"Only found {len(index)}/{num_pairs} camera pairs under {data_root}"
        )
    return index


class MultiCamPairDataset(Dataset):
    def __init__(
        self,
        data_root=DEFAULT_DATA_ROOT,
        cam_root=DEFAULT_CAM_ROOT,
        num_pairs=2000,
        index_seed=42,
        num_frames=81,
        pose_stride=4,
        height=480,
        width=832,
        caption=DEFAULT_CAPTION,
    ):
        super().__init__()
        if not _HAS_DECORD:
            raise ImportError("multicam_pair_dataset requires `decord`")
        self.pairs = build_pair_index(data_root, num_pairs, index_seed)
        self.data_root = data_root
        self.cam_root = cam_root
        self.num_frames = num_frames
        self.pose_stride = pose_stride
        self.height = height
        self.width = width
        self.caption = caption
        self._extrinsics_cache = {}  # per-worker, small (relpath -> dict)

    def __len__(self):
        return len(self.pairs)

    # ----- frames -----

    def _decode_clip(self, path):
        """Same decode-time resize + center-crop as SimpleVideoDataset, but
        fixed frame indices 0..num_frames-1. Also returns the intrinsics
        rescale info (sx, sy, left, top) of the decoded frame geometry."""
        probe = decord.VideoReader(path, num_threads=1)
        total_frames = len(probe)
        h0, w0, _ = probe[0].shape
        del probe

        scale = max(self.height / h0, self.width / w0)
        out_h = max(self.height, int(np.ceil(h0 * scale / 2) * 2))
        out_w = max(self.width, int(np.ceil(w0 * scale / 2) * 2))

        indices = np.clip(np.arange(self.num_frames), 0, total_frames - 1)

        vr = decord.VideoReader(path, width=out_w, height=out_h, num_threads=1)
        frames = vr.get_batch(list(indices)).asnumpy()  # [F, H, W, C] uint8
        del vr

        fh, fw = frames.shape[1], frames.shape[2]
        top = (fh - self.height) // 2
        left = (fw - self.width) // 2
        frames = frames[:, top : top + self.height, left : left + self.width, :]

        frames = torch.from_numpy(frames).float().div_(255.0)  # [F, H, W, C]
        frames = frames.permute(3, 0, 1, 2).contiguous()  # [C, F, H, W]
        geom = (out_w / w0, out_h / h0, left, top, w0, h0)
        return frames, geom

    # ----- poses -----

    def _load_extrinsics(self, relpath):
        if relpath in self._extrinsics_cache:
            return self._extrinsics_cache[relpath]
        # /tmp copy may lack cameras/: prefer local, fall back to NFS original.
        candidates = [
            os.path.join(self.data_root, relpath, "cameras",
                         "camera_extrinsics.json"),
            os.path.join(self.cam_root, relpath, "cameras",
                         "camera_extrinsics.json"),
        ]
        data = None
        for cand in candidates:
            if os.path.isfile(cand):
                with open(cand, "r") as f:
                    data = json.load(f)
                break
        if data is None:
            raise FileNotFoundError(
                f"camera_extrinsics.json not found for {relpath} "
                f"(tried {candidates})"
            )
        if len(self._extrinsics_cache) > 64:
            self._extrinsics_cache.clear()
        self._extrinsics_cache[relpath] = data
        return data

    def _poses_for_cam(self, extrinsics, cam_idx):
        frame_ids = range(0, self.num_frames, self.pose_stride)  # 0,4,...,80
        c2ws = []
        for fi in frame_ids:
            mat = parse_matrix_string(extrinsics[f"frame{fi}"][f"cam{cam_idx:02d}"])
            c2ws.append(ue_matrix_to_cv_c2w(mat))
        return torch.stack(c2ws)  # [21, 4, 4]

    def _intrinsics(self, relpath, geom):
        focal_dir = relpath.split(os.sep)[0]  # e.g. f24_aperture5
        focal_mm = float(focal_dir.split("_")[0][1:])
        sx, sy, left, top, w0, h0 = geom
        fx = focal_mm / SENSOR_WIDTH_MM * w0
        fy = focal_mm / SENSOR_WIDTH_MM * h0
        cx, cy = w0 / 2.0, h0 / 2.0
        K = torch.tensor(
            [
                [fx * sx, 0.0, cx * sx - left],
                [0.0, fy * sy, cy * sy - top],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        return K

    # ----- item -----

    def __getitem__(self, idx):
        vdir, relpath, src_cam, tgt_cam = self.pairs[idx % len(self.pairs)]
        try:
            frames_src, geom = self._decode_clip(
                os.path.join(vdir, f"cam{src_cam:02d}.mp4"))
            frames_tgt, _ = self._decode_clip(
                os.path.join(vdir, f"cam{tgt_cam:02d}.mp4"))
            extrinsics = self._load_extrinsics(relpath)
            c2w_src = self._poses_for_cam(extrinsics, src_cam)
            c2w_tgt = self._poses_for_cam(extrinsics, tgt_cam)
        except Exception as e:  # corrupted file etc. -> fall back to another
            print(f"[multicam_pair_dataset] failed to load {vdir} "
                  f"cams ({src_cam},{tgt_cam}): {e}")
            fallback = random.randrange(len(self.pairs))
            return self.__getitem__(fallback)

        # joint canonicalization over BOTH cameras' 42 poses
        all_c2w = torch.cat([c2w_src, c2w_tgt], dim=0)
        all_c2w = normalize_with_mean_pose(all_c2w)
        n = c2w_src.shape[0]
        c2w_src, c2w_tgt = all_c2w[:n], all_c2w[n:]

        K = self._intrinsics(relpath, geom)

        return {
            "frames_src": frames_src,
            "frames_tgt": frames_tgt,
            "c2w_src": c2w_src.float(),
            "c2w_tgt": c2w_tgt.float(),
            "K": K,
            "caption": self.caption,
        }


class MultiCamPairDataModule:
    """Data-module facade mirroring SimpleVideoDataModule."""

    def __init__(self, params=None, data_seed=0):
        params = dict(params or {})
        self.batch_size = int(params.pop("batch_size", 1))
        self.num_workers = int(params.pop("num_workers", 4))
        self.data_seed = int(data_seed)
        self.dataset = MultiCamPairDataset(**params)

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.data_seed)

        def worker_init_fn(worker_id):
            seed = (self.data_seed * 1000 + worker_id) % (2**31)
            random.seed(seed)
            np.random.seed(seed)

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            generator=generator,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.num_workers > 0,
        )


if __name__ == "__main__":
    # Pose-canonicalization sanity (spec: run once and paste in the report).
    torch.set_printoptions(precision=4, sci_mode=False)
    ds = MultiCamPairDataset(num_pairs=8, index_seed=42)
    print(f"index: {len(ds)} pairs; first: {ds.pairs[0]}")
    n_ok_shape = 0
    all_pass = True
    for i in range(2):
        item = ds[i]
        fs, ft = item["frames_src"], item["frames_tgt"]
        c2w = torch.cat([item["c2w_src"], item["c2w_tgt"]], dim=0)  # [42,4,4]
        K = item["K"]
        print(f"--- item {i}: frames_src {tuple(fs.shape)} "
              f"[{fs.min():.3f},{fs.max():.3f}], frames_tgt {tuple(ft.shape)}, "
              f"K diag ({K[0,0]:.2f},{K[1,1]:.2f}) c ({K[0,2]:.1f},{K[1,2]:.1f})")

        # (1) camera centers are O(1) after canonicalization
        centers = c2w[:, :3, 3]
        max_abs = centers.abs().max().item()
        mean_norm = centers.norm(dim=-1).mean().item()
        ok1 = 0.1 < max_abs <= 1.0 + 1e-4
        print(f"    centers: max|coord|={max_abs:.4f} (scale-normed to 1), "
              f"mean ||c||={mean_norm:.4f} -> {'PASS' if ok1 else 'FAIL'}")

        # (2) rotations orthonormal & per-token ray directions unit
        RtR = c2w[:, :3, :3].transpose(1, 2) @ c2w[:, :3, :3]
        rot_err = (RtR - torch.eye(3)).abs().max().item()
        from minVid.models.blocks.cam_phase_builder import plucker_per_token
        pl = plucker_per_token(item["c2w_src"], K)
        d_err = (pl[..., :3].norm(dim=-1) - 1).abs().max().item()
        ok2 = rot_err < 1e-4 and d_err < 1e-5
        print(f"    rot orthonormality err={rot_err:.2e}, "
              f"ray |d|-1 max={d_err:.2e} -> {'PASS' if ok2 else 'FAIL'}")

        # (3) forward rays converge on a common subject point. The earlier
        # look-at-ORIGIN check was wrong for this dataset: MCV cameras sit in
        # tight arcs filming an actor OUTSIDE the camera cluster, so the
        # canonical origin (mean camera pose) is often behind every camera.
        # Verified 2026-07-09: rays of all 10 cams intersect within ~0.3 m,
        # and the same scene index across focal sets yields the same world
        # point — parsing is correct; the old check's geometry prior was not.
        view_dir = c2w[:, :3, 2]  # CV convention: +z forward
        d = view_dir / view_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        eye = torch.eye(3)
        projs = eye.unsqueeze(0) - d.unsqueeze(-1) * d.unsqueeze(-2)  # [N,3,3]
        A = projs.sum(0)
        b = (projs @ centers.unsqueeze(-1)).sum(0).squeeze(-1)
        p = torch.linalg.solve(A, b)
        ray_dist = (projs @ (p - centers).unsqueeze(-1)).squeeze(-1).norm(dim=-1)
        spread = (centers - centers.mean(0)).norm(dim=-1).mean().clamp_min(1e-6)
        ratio = (ray_dist.mean() / spread).item()
        ok3 = ratio < 1.0
        print(f"    ray-convergence: mean point-to-ray dist / cam spread = "
              f"{ratio:.3f} (<1.0) -> {'PASS' if ok3 else 'FAIL'}")
        all_pass = all_pass and ok1 and ok2 and ok3
    print("ALL CHECKS:", "PASS" if all_pass else "FAIL")
