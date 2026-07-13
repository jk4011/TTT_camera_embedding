"""Q11 precompute: VAE latents for every unique (relpath, cam) video used by
the recam_ttt experiments (2000-pair train index, seed 42, + the 64-pair
holdout list), with ReCamMaster's TRAINING conventions:
  - official imageio decode + PIL cover-resize + CenterCrop to 480x832, [-1,1]
  - official Wan2.1 VAE, non-tiled encode (their data_process default), bf16

One .pt per video: {"latent": bf16 [16, 21, 60, 104]}. Existing files are
skipped, so reruns/resubmissions resume. Shard with --shard/--num_shards.

--emit_prompt additionally saves the shared caption embedding (and their
official negative prompt embedding, for later CFG evals) once: prompt_emb.pt.

Usage (envs/recam, from lact_ar_video):
  PYTHONPATH=.:/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/recam_ckpt/ReCamMaster \
  .../envs/recam/bin/python recam_ttt/precompute_latents.py \
      --out /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/mcv_latents_recam \
      --shard 0 --num_shards 6
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "minVid"))
from eval_recam_anchor import WAN_DIR, load_source_video_official  # noqa: E402

DATA_ROOT = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/MultiCamVideo-Dataset/train"
HOLDOUT_JSON = os.path.join(os.path.dirname(__file__), "..", "minVid",
                            "configs", "ar", "ccv_holdout_pairs_64.json")
CAPTION = "a person moving through a scene, cinematic camera"  # ccv fixed caption
NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def unique_videos():
    """Deterministic sorted list of (relpath, cam) across train + holdout."""
    from minVid.data.multicam_pair_dataset import MultiCamPairDataset
    ds = MultiCamPairDataset(
        data_root=DATA_ROOT, cam_root=DATA_ROOT,
        num_pairs=int(os.environ.get("PRE_NUM_PAIRS", "6000")), index_seed=42,
        num_frames=81, height=480, width=832, caption="x")
    uniq = set()
    for _, rel, s, t in ds.pairs:
        uniq.add((rel, int(s)))
        uniq.add((rel, int(t)))
    with open(HOLDOUT_JSON) as f:
        payload = json.load(f)
    for p in payload["pairs"]:
        uniq.add((p["relpath"], int(p["src_cam"])))
        uniq.add((p["relpath"], int(p["tgt_cam"])))
    return sorted(uniq)


def latent_path(out_dir, rel, cam):
    return os.path.join(out_dir, f"{rel.replace(os.sep, '_')}_cam{cam:02d}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--emit_prompt", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    order_file = os.environ.get("PRE_ORDER_FILE")
    if order_file:
        # priority order (training-stream first-need); striding preserves it
        with open(order_file) as f:
            vids = [(r, int(c)) for r, c in json.load(f)]
    else:
        vids = unique_videos()
    mine = vids[args.shard::args.num_shards]
    todo = [(r, c) for r, c in mine
            if not os.path.exists(latent_path(args.out, r, c))]
    print(f"[pre] shard {args.shard}/{args.num_shards}: {len(mine)} videos, "
          f"{len(todo)} to encode", flush=True)

    from diffsynth import ModelManager
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([os.path.join(WAN_DIR, "Wan2.1_VAE.pth")])
    vae = mm.fetch_model("wan_video_vae")
    vae.to(args.device)

    if args.emit_prompt:
        prompt_file = os.path.join(args.out, "prompt_emb.pt")
        if not os.path.exists(prompt_file):
            # one-time: build the full official pipeline (wires the wan
            # prompter + tokenizer correctly) just to embed two strings
            from eval_recam_anchor import build_recam_pipeline
            pipe = build_recam_pipeline(device=args.device)
            with torch.no_grad():
                pos = pipe.encode_prompt(CAPTION, positive=True)
                neg = pipe.encode_prompt(NEGATIVE, positive=False)
            pos = {k: v.cpu() for k, v in pos.items()}
            neg = {k: v.cpu() for k, v in neg.items()}
            torch.save({"caption": CAPTION, "prompt_emb": pos,
                        "negative": NEGATIVE, "negative_emb": neg}, prompt_file)
            print(f"[pre] saved {prompt_file}", flush=True)
            del pipe
            torch.cuda.empty_cache()

    for i, (rel, cam) in enumerate(todo):
        dst = latent_path(args.out, rel, cam)
        mp4 = os.path.join(DATA_ROOT, rel, "videos", f"cam{cam:02d}.mp4")
        tic = time.time()
        video = load_source_video_official(mp4)[None]  # [1,C,81,H,W] in [-1,1]
        with torch.no_grad():
            lat = vae.encode(
                video.to(dtype=torch.bfloat16, device=args.device),
                device=args.device, tiled=False)[0]
        tmp = dst + ".tmp"
        torch.save({"latent": lat.to(torch.bfloat16).cpu()}, tmp)
        os.replace(tmp, dst)
        if i % 20 == 0 or i == len(todo) - 1:
            print(f"[pre] {i + 1}/{len(todo)} {rel} cam{cam:02d} "
                  f"{tuple(lat.shape)} {time.time() - tic:.1f}s", flush=True)
    print(f"[pre] shard {args.shard} DONE", flush=True)


if __name__ == "__main__":
    main()
