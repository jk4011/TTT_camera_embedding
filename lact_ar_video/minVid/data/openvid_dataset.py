"""OpenVid-1M subset loader for the video2 (real-video, per-clip caption) task.

Reads an index json built by data_preprocess (list of {"file", "caption"}),
splits it deterministically (seeded shuffle, first n_train = train, the rest =
held-out val), and decodes clips exactly like SimpleVideoDataset (81 frames at
16 fps, resize + center crop to height x width).

Train split: the temporal crop start is random (data augmentation, as in
SimpleVideoDataset). Val split: the temporal crop is made deterministic per
clip so that different arms evaluate the same frames (paired evals).
"""
import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from minVid.data.simple_video_dataset import SimpleVideoDataset

SPLIT_SEED = 42


class OpenVidDataset(SimpleVideoDataset):

    def __init__(
        self,
        data_root,
        index_file,
        split="train",
        n_train=2000,
        num_frames=81,
        target_fps=16.0,
        height=480,
        width=832,
    ):
        Dataset.__init__(self)
        entries = json.load(open(index_file))
        entries = sorted(entries, key=lambda e: e["file"])
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(entries)
        assert split in ("train", "val")
        self.split = split
        self.entries = entries[:n_train] if split == "train" else entries[n_train:]
        assert len(self.entries) > 0, f"empty {split} split ({len(entries)} total)"
        self.video_paths = [os.path.join(data_root, e["file"]) for e in self.entries]
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.height = height
        self.width = width

    def _decode_clip(self, path):
        if self.split == "val":
            # freeze python-random so the temporal crop start is a pure
            # function of the clip -> identical frames across arms/processes
            st = random.getstate()
            random.seed(hash(os.path.basename(path)) & 0x7FFFFFFF)
            try:
                return super()._decode_clip(path)
            finally:
                random.setstate(st)
        return super()._decode_clip(path)

    def __getitem__(self, idx):
        i = idx % len(self.video_paths)
        try:
            frames = self._decode_clip(self.video_paths[i])
        except Exception as e:
            print(f"[openvid_dataset] failed to decode {self.video_paths[i]}: {e}")
            return self.__getitem__(random.randrange(len(self.video_paths)))
        return {"frames": frames, "caption": self.entries[i]["caption"]}


class OpenVidDataModule:
    """Same facade as SimpleVideoDataModule (see get_data_module)."""

    def __init__(self, params=None, data_seed=0):
        params = dict(params or {})
        self.batch_size = int(params.pop("batch_size", 1))
        self.num_workers = int(params.pop("num_workers", 4))
        self.data_seed = int(data_seed)
        self.dataset = OpenVidDataset(**params)

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
