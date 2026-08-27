"""video2 (unconditional AR video) evaluation: paired held-out validation loss.

Computes the EXACT training objective (VideoLatentFlowMatching.forward: AR
interleave flow-matching MSE with logit-normal weighting) forward-only on a
FIXED held-out clip list, with per-clip deterministic noise and diffusion
timesteps (torch.manual_seed keyed on the clip's position in the list). Two
invocations on the same checkpoint yield identical numbers; different arms
evaluated on the same list are paired per (clip, noise, t) for paired-t
analysis.

Held-out selection: the training run draws its clips from
build_video_index(data_root, num_clips, index_seed) (seed 42). The val list is
build_video_index(..., index_seed=43) minus the training set, so no evaluated
clip was ever trained on.

Usage (from lact_ar_video/minVid, PYTHONPATH=<repo>/lact_ar_video):
  python eval_video2_valloss.py --config configs/ar/video2_t5_base.yaml \
      --ckpt ../outputs/video2_t5_base/seed_1/checkpoint_model_001499 \
      --out ../outputs/eval_dev/valloss_video2_t5_base_1499.json
"""
import argparse
import json
import os
import time

import eval_ccv_common as common  # noqa: F401  (sets env before torch use)
import torch

from eval_ccv_common import (
    cache_and_free_text_encoder,
    load_config,
    load_model,
)

CLIP_SEED_BASE = 555000  # per-clip seed = CLIP_SEED_BASE + clip index


def main():
    parser = argparse.ArgumentParser(description="video2 held-out paired val loss")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint_model_XXXXXX dir (contains dcp/)")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n_clips", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    cfg = load_config(args.config)
    model = load_model(cfg, args.ckpt, args.device)

    dp = cfg.dataset_train.params
    caption = dp.get("caption", "")
    model = cache_and_free_text_encoder(model, [caption], args.device)

    from minVid.data.simple_video_dataset import SimpleVideoDataset, build_video_index

    train_paths = set(build_video_index(dp.data_root, dp.num_clips, dp.index_seed))
    cand = build_video_index(dp.data_root, dp.num_clips + 4 * args.n_clips, 43)
    val_paths = [p for p in cand if p not in train_paths][: args.n_clips]
    assert len(val_paths) == args.n_clips, \
        f"only {len(val_paths)} held-out clips available"

    ds_params = {k: v for k, v in dict(dp).items()
                 if k not in ("batch_size", "num_workers")}
    ds_params["num_clips"] = 1  # cheap index build, then overridden
    dataset = SimpleVideoDataset(**ds_params)
    dataset.video_paths = val_paths

    records = []
    t0 = time.time()
    for i in range(args.n_clips):
        item = dataset[i]
        batch = {
            # [C, F, H, W] -> [1, F, C, H, W]
            "video_rgb": item["frames"].permute(1, 0, 2, 3)[None].to(args.device),
            "text_prompts": [caption],
        }
        torch.manual_seed(CLIP_SEED_BASE + i)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            rd = model(batch)
        records.append({"clip": os.path.relpath(val_paths[i], dp.data_root),
                        "loss": float(rd["loss"])})
        if (i + 1) % 8 == 0:
            print(f"  {i + 1}/{args.n_clips} mean so far "
                  f"{sum(r['loss'] for r in records) / len(records):.6f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    mean = sum(r["loss"] for r in records) / len(records)
    out = {
        "config": args.config,
        "ckpt": args.ckpt,
        "n_clips": args.n_clips,
        "clip_seed_base": CLIP_SEED_BASE,
        "mean_loss": mean,
        "records": records,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"DONE mean_loss {mean:.6f} over {args.n_clips} clips -> {args.out}")


if __name__ == "__main__":
    main()
