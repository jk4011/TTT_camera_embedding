"""Shared plumbing for the ccv (camera-controlled video) evaluation scripts.

Provides:
  - environment defaults (noexec /tmp: triton/inductor caches on lustre),
  - config loading (same OmegaConf+edict path as train.py),
  - model construction + DCP checkpoint loading WITHOUT FSDP / process group
    (the training DCP dirs hold the full generator state dict with keys
    relative to WanDiffusionWrapper, EMA weights already copied in by
    save_job),
  - a cached text-encoder stub (the ccv caption is a single fixed string, so
    the 5.7B umt5-xxl is used once and freed),
  - construction of the FIXED held-out pair list: scenes that never appear in
    the training pair index (build_pair_index(data_root, num_pairs=2000,
    index_seed=42) -- identical across all 6 ccv runs), one (src, tgt) camera
    pair per held-out scene, saved as json so every evaluation uses the
    identical list,
  - per-frame SSIM (no skimage in .venv_llm).

Run from lact_ar_video/minVid with PYTHONPATH=<repo>/lact_ar_video.
"""
import json
import os
import random

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def setup_env():
    """Cache/env defaults for compiled kernels on the noexec-/tmp node."""
    defaults = {
        "TRITON_CACHE_DIR": os.path.join(_REPO_ROOT, ".cache_triton"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(_REPO_ROOT, ".cache_inductor"),
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "HF_HOME": "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/hf_cache",
        # lpips pulls the AlexNet backbone via torch.hub; keep it on lustre
        "TORCH_HOME": "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/torch_home",
        "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
        "TRITON_CUOBJDUMP_PATH": "/usr/local/cuda/bin/cuobjdump",
        "TRITON_NVDISASM_PATH": "/usr/local/cuda/bin/nvdisasm",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)


setup_env()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from easydict import EasyDict as edict  # noqa: E402

DEFAULT_PAIRS_JSON = os.path.join(
    os.path.dirname(__file__), "configs", "ar", "ccv_holdout_pairs_64.json"
)
HOLDOUT_SEED = 20260711
DEFAULT_N_HOLDOUT = 64


# --------------------------------------------------------------------------
# config / model / checkpoint
# --------------------------------------------------------------------------

def load_config(config_path):
    config = OmegaConf.load(config_path)
    config = OmegaConf.create(config)
    return edict(config)


def load_model(config, ckpt_dir, device="cuda"):
    """Build VideoLatentFlowMatching from the training config and load the
    DCP checkpoint into its generator (WanDiffusionWrapper).

    ckpt_dir: .../checkpoint_model_013999 (containing dcp/) or the dcp dir.
    The DCP was saved by save_job AFTER ema_params.copy_to_model(), i.e. the
    stored weights are the EMA weights.
    """
    from minVid.utils.config_utils import instantiate_from_config
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    model = instantiate_from_config(config.model)
    model.eval()
    model.requires_grad_(False)

    dcp_dir = ckpt_dir
    if os.path.isdir(os.path.join(ckpt_dir, "dcp")):
        dcp_dir = os.path.join(ckpt_dir, "dcp")
    assert os.path.isdir(dcp_dir), f"DCP dir not found: {dcp_dir}"

    reader = dist_cp.FileSystemReader(dcp_dir)
    ckpt_keys = set(reader.read_metadata().state_dict_metadata.keys())

    sd = model.generator.state_dict()
    model_keys = set(sd.keys())
    missing_in_ckpt = sorted(model_keys - ckpt_keys)
    extra_in_ckpt = sorted(ckpt_keys - model_keys)
    if missing_in_ckpt:
        print(f"[eval_ccv] WARNING: {len(missing_in_ckpt)} model tensors not in "
              f"checkpoint (keep init): {missing_in_ckpt[:8]} ...")
    if extra_in_ckpt:
        print(f"[eval_ccv] WARNING: {len(extra_in_ckpt)} checkpoint tensors not in "
              f"model (ignored): {extra_in_ckpt[:8]} ...")

    load_sd = {k: v for k, v in sd.items() if k in ckpt_keys}
    dist_cp.load(
        state_dict=load_sd,
        storage_reader=reader,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    sd.update(load_sd)
    model.generator.load_state_dict(sd, strict=True)
    print(f"[eval_ccv] loaded {len(load_sd)}/{len(sd)} generator tensors from {dcp_dir}")

    model.to(device)
    return model


class CachedTextEncoder(nn.Module):
    """Drop-in stand-in for WanTextEncoder over a fixed set of prompts."""

    def __init__(self, cache):
        super().__init__()
        self.cache = {k: v.clone() for k, v in cache.items()}  # str -> [L, D]

    def forward(self, text_prompts):
        embeds = torch.stack([self.cache[p] for p in text_prompts], dim=0)
        return {"prompt_embeds": embeds}


def cache_and_free_text_encoder(model, prompts, device="cuda"):
    """Encode `prompts` once with the real umt5-xxl, then replace
    model.text_encoder with a cached stub (frees ~22 GB fp32)."""
    import gc

    model.text_encoder.to(device)
    cache = {}
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for p in prompts:
            out = model.text_encoder([p])["prompt_embeds"]  # [1, L, D]
            cache[p] = out[0].detach().to(device)
    model.text_encoder = CachedTextEncoder(cache).to(device)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[eval_ccv] cached text embeddings for {len(cache)} prompt(s); "
          f"umt5-xxl freed")
    return model


# --------------------------------------------------------------------------
# held-out pairs
# --------------------------------------------------------------------------

def build_holdout_pairs(dataset_params, n_pairs=DEFAULT_N_HOLDOUT,
                        holdout_seed=HOLDOUT_SEED):
    """One (src, tgt) camera pair per scene, scenes DISJOINT from the training
    pair index (same data_root / num_pairs / index_seed as the ccv configs)."""
    from minVid.data.multicam_pair_dataset import build_pair_index, NUM_CAMS

    data_root = dataset_params["data_root"]
    train_num_pairs = int(dataset_params.get("num_pairs", 2000))
    train_index_seed = int(dataset_params.get("index_seed", 42))

    train_index = build_pair_index(data_root, train_num_pairs, train_index_seed)
    train_scenes = {rel for (_vdir, rel, _s, _t) in train_index}

    all_scenes = []  # (videos_dir, relpath), sorted deterministic order
    for focal_dir in sorted(os.listdir(data_root)):
        focal_path = os.path.join(data_root, focal_dir)
        if not os.path.isdir(focal_path):
            continue
        for scene in sorted(os.listdir(focal_path)):
            rel = os.path.join(focal_dir, scene)
            vdir = os.path.join(focal_path, scene, "videos")
            all_scenes.append((vdir, rel))

    holdout_scenes = [(v, r) for (v, r) in all_scenes if r not in train_scenes]
    print(f"[eval_ccv] scenes: total {len(all_scenes)}, in training index "
          f"{len(train_scenes)}, held-out {len(holdout_scenes)}")
    assert len(holdout_scenes) >= n_pairs, \
        f"only {len(holdout_scenes)} held-out scenes for {n_pairs} pairs"

    rng = random.Random(holdout_seed)
    scenes = holdout_scenes[:]
    rng.shuffle(scenes)

    pairs = []
    for vdir, rel in scenes:
        if len(pairs) >= n_pairs:
            break
        cam_json = os.path.join(data_root, rel, "cameras",
                                "camera_extrinsics.json")
        if not os.path.isfile(cam_json):
            continue
        src_cam = rng.randint(1, NUM_CAMS)
        tgt_cam = rng.randint(1, NUM_CAMS - 1)
        if tgt_cam >= src_cam:
            tgt_cam += 1
        src_path = os.path.join(vdir, f"cam{src_cam:02d}.mp4")
        tgt_path = os.path.join(vdir, f"cam{tgt_cam:02d}.mp4")
        if not (os.path.isfile(src_path) and os.path.isfile(tgt_path)):
            continue
        pairs.append({
            "videos_dir": vdir,
            "relpath": rel,
            "src_cam": src_cam,
            "tgt_cam": tgt_cam,
        })
    assert len(pairs) == n_pairs, f"found only {len(pairs)}/{n_pairs} pairs"
    return {
        "description": "ccv held-out evaluation pairs (scenes disjoint from "
                       "the training pair index)",
        "data_root": data_root,
        "train_num_pairs": train_num_pairs,
        "train_index_seed": train_index_seed,
        "holdout_seed": holdout_seed,
        "n_pairs": n_pairs,
        "pairs": pairs,
    }


def load_or_build_pairs(pairs_path, config, n_pairs=DEFAULT_N_HOLDOUT):
    """Load the fixed pair list; build+save it (deterministic) if missing."""
    if os.path.isfile(pairs_path):
        with open(pairs_path, "r") as f:
            payload = json.load(f)
        print(f"[eval_ccv] loaded {len(payload['pairs'])} held-out pairs from "
              f"{pairs_path}")
        return payload
    dataset_params = config.dataset_train.params
    payload = build_holdout_pairs(dataset_params, n_pairs=n_pairs)
    os.makedirs(os.path.dirname(os.path.abspath(pairs_path)), exist_ok=True)
    with open(pairs_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[eval_ccv] built and saved {len(payload['pairs'])} held-out pairs "
          f"to {pairs_path}")
    return payload


def make_pair_dataset(config, pairs):
    """MultiCamPairDataset restricted to the given explicit pair list."""
    from minVid.data.multicam_pair_dataset import MultiCamPairDataset

    params = dict(config.dataset_train.params)
    params.pop("batch_size", None)
    params.pop("num_workers", None)
    params["num_pairs"] = 8  # cheap ctor; pairs overridden below
    ds = MultiCamPairDataset(**params)
    ds.pairs = [
        (p["videos_dir"], p["relpath"], p["src_cam"], p["tgt_cam"])
        for p in pairs
    ]
    return ds


def pair_batch_to_gpu(item, device):
    """Mirror train.py's ccv batch construction for a single dataset item."""
    return {
        "video_rgb_src": item["frames_src"][None].permute(0, 2, 1, 3, 4).to(device),
        "video_rgb_tgt": item["frames_tgt"][None].permute(0, 2, 1, 3, 4).to(device),
        "c2w_src": item["c2w_src"][None].to(device),
        "c2w_tgt": item["c2w_tgt"][None].to(device),
        "K": item["K"][None].to(device),
        "text_prompts": [item["caption"]],
    }


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _gaussian_window(size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]  # [size, size]


@torch.no_grad()
def ssim_per_frame(pred, gt, data_range=1.0):
    """Standard gaussian-window SSIM, averaged over channels+pixels per frame.

    pred/gt: [F, C, H, W] float in [0, data_range]. Returns [F] tensor.
    """
    F_, C, _, _ = pred.shape
    win = _gaussian_window(11, 1.5, pred.device)[None, None].expand(C, 1, 11, 11)
    pad = 11 // 2
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    def filt(x):
        return torch.nn.functional.conv2d(x, win, padding=pad, groups=C)

    pred = pred.float()
    gt = gt.float()
    mu_p, mu_g = filt(pred), filt(gt)
    mu_p2, mu_g2, mu_pg = mu_p * mu_p, mu_g * mu_g, mu_p * mu_g
    sigma_p2 = filt(pred * pred) - mu_p2
    sigma_g2 = filt(gt * gt) - mu_g2
    sigma_pg = filt(pred * gt) - mu_pg
    ssim_map = ((2 * mu_pg + C1) * (2 * sigma_pg + C2)) / (
        (mu_p2 + mu_g2 + C1) * (sigma_p2 + sigma_g2 + C2)
    )
    return ssim_map.mean(dim=(1, 2, 3))


@torch.no_grad()
def psnr_per_frame(pred, gt, data_range=1.0):
    """pred/gt: [F, C, H, W] float. Returns [F] tensor (dB)."""
    mse = ((pred.float() - gt.float()) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(data_range ** 2 / mse)


if __name__ == "__main__":
    # Standalone builder for the fixed held-out pair list.
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Build the fixed ccv held-out pair list")
    parser.add_argument("--config", type=str,
                        default="configs/ar/abl_ccv_base.yaml")
    parser.add_argument("--out", type=str, default=DEFAULT_PAIRS_JSON)
    parser.add_argument("--n_pairs", type=int, default=DEFAULT_N_HOLDOUT)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if os.path.isfile(args.out):
        print(f"pair list already exists: {args.out} (delete to rebuild)")
    else:
        load_or_build_pairs(args.out, cfg, n_pairs=args.n_pairs)
