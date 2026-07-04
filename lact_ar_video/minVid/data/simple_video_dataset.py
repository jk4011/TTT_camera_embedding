"""
Minimal video dataset + data module for the lact_ar_video ablation runs.

Reads random clips from the MultiCamVideo dataset
(<root>/<f*_aperture*>/<sceneN>/videos/camXX.mp4, UE5-rendered,
81 frames @ 15fps, 1280x1280) and returns:

    {
        "frames":  float32 tensor [C, F, H, W] in [0, 1],
        "caption": str,
    }

train.py accesses batches as data_batch["frames"] with shape
[B, C, F, H, W] (default collate) and data_batch["caption"] as a
list of strings, so the default pytorch collate works.

The data module class is instantiated via minVid.data.get_data_module()
with (params_dict, data_seed=...) and must expose .train_dataloader().
"""

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


DEFAULT_DATA_ROOT = (
    "/NHNHOME/WORKSPACE/26msit001_T_B/POSTECH-CGLAB/dataset/"
    "MultiCamVideo-Dataset/train"
)
DEFAULT_CAPTION = "a person moving through a scene, cinematic camera"


def build_video_index(data_root, num_clips=2000, index_seed=42):
    """Build a fixed, seeded index of `num_clips` video paths.

    Enumerates <root>/<focal_dir>/<scene>/videos/cam*.mp4 lazily:
    we list the (comparatively few) scene directories, then sample
    (scene, camera) pairs with a fixed seed and keep the ones that exist.
    Deterministic for a given (data_root, num_clips, index_seed).
    """
    rng = random.Random(index_seed)

    scene_video_dirs = []
    for focal_dir in sorted(os.listdir(data_root)):
        focal_path = os.path.join(data_root, focal_dir)
        if not os.path.isdir(focal_path):
            continue
        for scene in sorted(os.listdir(focal_path)):
            vdir = os.path.join(focal_path, scene, "videos")
            scene_video_dirs.append(vdir)

    if len(scene_video_dirs) == 0:
        raise RuntimeError(f"No scene directories found under {data_root}")

    index = []
    seen = set()
    max_attempts = num_clips * 20
    attempts = 0
    while len(index) < num_clips and attempts < max_attempts:
        attempts += 1
        vdir = rng.choice(scene_video_dirs)
        cam = rng.randint(1, 10)
        path = os.path.join(vdir, f"cam{cam:02d}.mp4")
        if path in seen:
            continue
        if os.path.isfile(path):
            seen.add(path)
            index.append(path)

    if len(index) < num_clips:
        raise RuntimeError(
            f"Only found {len(index)}/{num_clips} videos under {data_root}"
        )
    return index


class SimpleVideoDataset(Dataset):
    """Decodes a contiguous `num_frames`-frame clip from a random mp4.

    Frames are decode-time resized so that the short side covers the
    target, then center-cropped to (height, width). Output is float32
    in [0, 1], layout [C, F, H, W].
    """

    def __init__(
        self,
        data_root=DEFAULT_DATA_ROOT,
        num_clips=2000,
        index_seed=42,
        num_frames=81,
        target_fps=16.0,
        height=480,
        width=832,
        caption=DEFAULT_CAPTION,
    ):
        super().__init__()
        if not _HAS_DECORD:
            raise ImportError("simple_video_dataset requires `decord`")
        self.video_paths = build_video_index(data_root, num_clips, index_seed)
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.height = height
        self.width = width
        self.caption = caption

    def __len__(self):
        return len(self.video_paths)

    def _decode_clip(self, path):
        # Probe native resolution / fps / frame count (decodes one frame).
        probe = decord.VideoReader(path, num_threads=1)
        total_frames = len(probe)
        native_fps = float(probe.get_avg_fps()) or self.target_fps
        h0, w0, _ = probe[0].shape
        del probe

        # Decode-time resize so both target dims are covered.
        scale = max(self.height / h0, self.width / w0)
        # decord requires even dims for some codecs; round up to even.
        out_h = max(self.height, int(np.ceil(h0 * scale / 2) * 2))
        out_w = max(self.width, int(np.ceil(w0 * scale / 2) * 2))

        # Frame stride to approximate target_fps (15fps native -> stride 1).
        stride = max(1, int(round(native_fps / self.target_fps)))
        span = (self.num_frames - 1) * stride + 1
        if total_frames >= span:
            start = random.randint(0, total_frames - span)
        else:
            start = 0
        indices = start + stride * np.arange(self.num_frames)
        # Clamp (repeats last frame if the video is shorter than needed).
        indices = np.clip(indices, 0, total_frames - 1)

        vr = decord.VideoReader(path, width=out_w, height=out_h, num_threads=1)
        frames = vr.get_batch(list(indices)).asnumpy()  # [F, H, W, C] uint8
        del vr

        # Center crop to (height, width).
        fh, fw = frames.shape[1], frames.shape[2]
        top = (fh - self.height) // 2
        left = (fw - self.width) // 2
        frames = frames[:, top : top + self.height, left : left + self.width, :]

        frames = torch.from_numpy(frames).float().div_(255.0)  # [F, H, W, C]
        frames = frames.permute(3, 0, 1, 2).contiguous()  # [C, F, H, W]
        return frames

    def __getitem__(self, idx):
        path = self.video_paths[idx % len(self.video_paths)]
        try:
            frames = self._decode_clip(path)
        except Exception as e:  # corrupted file etc. -> fall back to another
            print(f"[simple_video_dataset] failed to decode {path}: {e}")
            fallback = random.randrange(len(self.video_paths))
            return self.__getitem__(fallback)
        return {"frames": frames, "caption": self.caption}


class SimpleVideoDataModule:
    """Minimal data-module facade expected by minVid/train.py."""

    def __init__(self, params=None, data_seed=0):
        params = dict(params or {})
        self.batch_size = int(params.pop("batch_size", 1))
        self.num_workers = int(params.pop("num_workers", 4))
        self.data_seed = int(data_seed)
        self.dataset = SimpleVideoDataset(**params)

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
