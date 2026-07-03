"""RE10K dataset reading per-scene .torch files produced by data_preprocess/reshard_re10k.py.

Scene dict: {url, timestamps [N], cameras [N, 18], images: list of N jpeg-byte uint8 tensors, key}
cameras row: [fx fy cx cy 0 0, w2c(3x4) flattened], intrinsics normalized by image size.
"""
import json
import os
import random

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms.v2 import functional as TF

from data import normalize_with_mean_pose


def decode_resize_crop(jpeg_bytes, target_size):
    """Decode jpeg bytes -> resize (cover) -> center crop to target_size (h, w).

    Returns:
        image: float tensor [3, h, w] in [0, 1]
        (resized_h, resized_w, left, top): resize/crop geometry for intrinsics
    """
    img = decode_image(jpeg_bytes)  # [c, H, W] uint8
    _, orig_h, orig_w = img.shape
    target_h, target_w = target_size

    scale = max(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    img = TF.resize(img, [new_h, new_w], antialias=True)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img[:, top : top + target_h, left : left + target_w]

    if img.size(0) == 1:
        img = img.expand(3, -1, -1)
    return img.float() / 255.0, (new_h, new_w, left, top)


class Re10KDataset(Dataset):
    def __init__(
        self,
        index_path,
        num_views,
        image_size,
        scene_pose_normalize=True,
        window=192,
        min_frames=None,
        eval_mode=False,
        num_input_views=None,
        num_target_views=None,
        max_scenes=None,
    ):
        """
        index_path: json list of {"file", "num_frames"}; scene files live in the
            sibling dir named like the index (e.g. train_index.json -> train/).
        eval_mode: deterministic view selection; returns
            [num_input_views inputs, num_target_views targets] in order.
        """
        self.base_dir = os.path.join(
            os.path.dirname(index_path),
            os.path.basename(index_path).replace("_index.json", ""),
        )
        entries = json.load(open(index_path, "r"))

        self.num_views = num_views
        min_frames = min_frames if min_frames is not None else num_views * 3
        self.entries = [e for e in entries if e["num_frames"] >= min_frames]

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = tuple(image_size)

        self.scene_pose_normalize = scene_pose_normalize
        self.window = window
        self.eval_mode = eval_mode
        self.num_input_views = num_input_views
        self.num_target_views = num_target_views
        if eval_mode:
            assert num_input_views is not None and num_target_views is not None
            assert num_input_views + num_target_views == num_views
        if max_scenes is not None:
            self.entries = self.entries[:max_scenes]

    def __len__(self):
        return len(self.entries)

    def _select_indices(self, num_frames):
        if self.eval_mode:
            # Deterministic: centered window; inputs uniformly spaced; targets
            # at window midpoints between inputs (shifted off collisions).
            w = min(num_frames, self.window)
            start = (num_frames - w) // 2
            n_in, n_tg = self.num_input_views, self.num_target_views
            inputs = [start + round(j * (w - 1) / (n_in - 1)) for j in range(n_in)]
            targets = []
            for j in range(n_tg):
                t = start + round((j + 0.5) * (w - 1) / n_tg)
                while t in inputs or t in targets:
                    t += 1
                targets.append(min(t, start + w - 1))
            return inputs + targets

        # Train: random contiguous window, random distinct views, random order.
        w = min(num_frames, random.randint(self.num_views * 3, self.window))
        start = random.randint(0, num_frames - w)
        return random.sample(range(start, start + w), self.num_views)

    def __getitem__(self, index):
        entry = self.entries[index]
        scene = torch.load(
            os.path.join(self.base_dir, entry["file"]),
            weights_only=False, map_location="cpu",
        )
        cameras = scene["cameras"]  # [N, 18]
        num_frames = len(scene["images"])
        indices = self._select_indices(num_frames)

        fxfycxcy_list, c2w_list, image_list = [], [], []
        for i in indices:
            image, (new_h, new_w, left, top) = decode_resize_crop(
                scene["images"][i], self.image_size
            )

            # Intrinsics are normalized by image size, so after resizing the
            # full image to (new_h, new_w): fx_pix = fx * new_w, etc. The
            # center crop then shifts the principal point.
            fx, fy, cx, cy = (cameras[i][j].item() for j in range(4))
            fxfycxcy_list.append([
                fx * new_w,
                fy * new_h,
                cx * new_w - left,
                cy * new_h - top,
            ])

            w2c = torch.eye(4)
            w2c[:3] = cameras[i][6:18].reshape(3, 4)
            c2w_list.append(torch.inverse(w2c))
            image_list.append(image)

        c2ws = torch.stack(c2w_list)
        if self.scene_pose_normalize:
            c2ws = normalize_with_mean_pose(c2ws)

        return {
            "fxfycxcy": torch.tensor(fxfycxcy_list),
            "c2w": c2ws,
            "image": torch.stack(image_list),
        }
