"""Q11 Phase-2: generation eval for a recam_ttt checkpoint with the OFFICIAL
ReCamMaster sampler (50 steps, sigma_shift 5.0, CFG 5.0, their negative
prompt, seed 0) — the same protocol as the external anchor
(eval_recam_anchor.py), so the numbers are directly comparable row-for-row.

The patched DiT (per-video attention + TTT branch) is a drop-in for their
pipeline __call__: the block signature is unchanged, only the internals
differ. The per-step camera/phase context is set once per pair (the phases
depend only on the pair, not the denoise step).

Usage (envs/recam, from lact_ar_video):
  PYTHONPATH=.:<ReCamMaster repo> python recam_ttt/eval_generate.py \
      --config recam_ttt/configs/h.yaml \
      --ckpt outputs/recam_ttt/h/ckpt_step3250.pt \
      --out outputs/recam_ttt/gen_h_3250 --n_pairs 8
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "minVid"))

import eval_ccv_common as common  # noqa: F401,E402  (sets env before torch use)
from eval_ccv_common import DEFAULT_PAIRS_JSON, load_config, make_pair_dataset  # noqa: E402
from eval_recam_anchor import (  # noqa: E402
    RECAM_NEGATIVE_PROMPT,
    build_cam_embedding,
    compute_metrics,
    load_source_video_official,
    to_uint8_video,
)
from recam_ttt.ttt_adapter import build_recam_coords6, set_ctx_phases  # noqa: E402
from recam_ttt.train_recam_ttt import build_patched_pipe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="recam_ttt variant yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=str, default=DEFAULT_PAIRS_JSON)
    ap.add_argument("--gt_config", type=str,
                    default="minVid/configs/ar/abl_ccv_base.yaml",
                    help="ccv config for the GT dataset params + caption")
    ap.add_argument("--n_pairs", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = args.device

    pipe, ttt_ctx, _ = build_patched_pipe(cfg, device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    named = dict(pipe.dit.named_parameters())
    for n, t in state["model_trainable"].items():
        named[n].data.copy_(t.to(named[n].dtype))
    print(f"[gen] loaded trainables from {args.ckpt} (step {state['step']})",
          flush=True)
    pipe.dit.eval()
    # generation needs the text encoder (negative prompt) and VAE back on GPU
    pipe.text_encoder.to(device)
    pipe.vae.to(device)

    gt_config = load_config(args.gt_config)
    caption = gt_config.dataset_train.params.caption

    with open(args.pairs) as f:
        payload = json.load(f)
    pairs = payload["pairs"][args.start: args.start + args.n_pairs]
    data_root = payload["data_root"]

    import lpips
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    ds = make_pair_dataset(gt_config, pairs)
    os.makedirs(args.out, exist_ok=True)

    need_coords = bool(cfg.ttt_input_rope) or bool(cfg.ttt_hidden_rope)
    results = []
    for i, pair in enumerate(pairs):
        rel, src_cam, tgt_cam = (pair["relpath"], int(pair["src_cam"]),
                                 int(pair["tgt_cam"]))
        rec = {"index": args.start + i, "relpath": rel, "src_cam": src_cam,
               "tgt_cam": tgt_cam, "seed": args.seed}
        print(f"[gen] pair {i}/{len(pairs)}: {rel} cams({src_cam},{tgt_cam})",
              flush=True)

        src_mp4 = os.path.join(pair["videos_dir"], f"cam{src_cam:02d}.mp4")
        cam_json = os.path.join(data_root, rel, "cameras",
                                "camera_extrinsics.json")
        source_video = load_source_video_official(src_mp4)[None]
        target_camera = build_cam_embedding(cam_json, cond_cam=src_cam,
                                            tgt_cam=tgt_cam)[None]
        coords6 = build_recam_coords6(cam_json, src_cam, tgt_cam, rel,
                                      device=device) if need_coords else None
        set_ctx_phases(ttt_ctx, coords6, pipe.dit.blocks[0].ttt_branch)

        tic = time.time()
        frames = pipe(
            prompt=[caption],
            negative_prompt=RECAM_NEGATIVE_PROMPT,
            source_video=source_video,
            target_camera=target_camera,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.steps,
            seed=args.seed, tiled=True,
        )
        rec["gen_seconds"] = time.time() - tic

        gen = torch.stack(
            [torch.from_numpy(np.array(fr)) for fr in frames]
        ).float().div_(255.0).permute(0, 3, 1, 2).to(device)
        item = ds[i]
        gt = item["frames_tgt"].permute(1, 0, 2, 3).to(device)
        n_f = min(gen.shape[0], gt.shape[0])
        gen, gt = gen[:n_f], gt[:n_f]
        rec.update(compute_metrics(gen, gt, lpips_model, device))
        results.append(rec)
        print(f"[gen]   PSNR {rec['psnr_mean']:.3f}  SSIM {rec['ssim_mean']:.4f}"
              f"  LPIPS {rec['lpips_mean']:.4f} ({rec['gen_seconds']:.0f}s)",
              flush=True)

        from minVid.utils.io_utils import save_video
        tag = (f"pair{rec['index']:03d}_{rel.replace(os.sep, '_')}"
               f"_cam{src_cam:02d}to{tgt_cam:02d}")
        gen_u8 = to_uint8_video(gen)
        gt_u8 = to_uint8_video(gt)
        src_u8 = to_uint8_video(item["frames_src"].permute(1, 0, 2, 3)[:n_f])
        save_video(gen_u8, os.path.join(args.out, f"{tag}_gen.mp4"),
                   save_fps=15)
        save_video(torch.cat([src_u8, gen_u8, gt_u8], dim=3),
                   os.path.join(args.out, f"{tag}_src_gen_gt.mp4"),
                   save_fps=15)
        _flush(args, cfg, state, results)

    _flush(args, cfg, state, results)
    print(f"[gen] SUMMARY {cfg.variant} over {len(results)} pairs: "
          f"PSNR {sum(r['psnr_mean'] for r in results)/len(results):.3f}  "
          f"SSIM {sum(r['ssim_mean'] for r in results)/len(results):.4f}  "
          f"LPIPS {sum(r['lpips_mean'] for r in results)/len(results):.4f}",
          flush=True)


def _flush(args, cfg, state, results):
    if not results:
        return
    out = {
        "config": os.path.abspath(args.config),
        "variant": str(cfg.variant),
        "ckpt": os.path.abspath(args.ckpt),
        "ckpt_step": int(state["step"]),
        "pairs_file": os.path.abspath(args.pairs),
        "sampler": {"type": "official_recammaster_flowmatch",
                    "steps": args.steps, "sigma_shift": 5.0,
                    "cfg_scale": args.cfg_scale, "seed": args.seed,
                    "negative_prompt": "official (Chinese)",
                    "tiled_vae": True},
        "n_pairs": len(results),
        "psnr_mean": sum(r["psnr_mean"] for r in results) / len(results),
        "ssim_mean": sum(r["ssim_mean"] for r in results) / len(results),
        "lpips_mean": sum(r["lpips_mean"] for r in results) / len(results),
        "per_pair": results,
    }
    path = os.path.join(args.out, "metrics.json")
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
