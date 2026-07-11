"""Phase-2 ccv evaluation: conditioned AR generation + GT-pixel metrics.

For each held-out (scene, srcCam, tgtCam) pair:
  1. VAE-encode the SRC video (clean latents, 21 latent frames).
  2. AR-generate the TGT video chunk by chunk (7 chunks x 3 latent frames)
     with an Euler flow-matching sampler, using the SAME timestep shift the
     training objective used (adjust_timestep_shift -> 2.1135 at 4680 tokens).
  3. Decode, save mp4s, compute per-frame PSNR / SSIM / LPIPS vs the GT
     target-camera video.

Sampler design (deviation from a stateful incremental pipeline, on purpose):
every denoise step runs the TRAINING forward on the full
[SRC clean 7 chunks || TGT interleave 13 slots] fake-batch sequence, with
  - SRC slots     = clean source latents (written into fast weights by the
                    kernel's src-prefix path, exactly as in training),
  - clean slots j = already-generated chunks j < i,
  - noisy slot i  = current x_sigma with timestep sigma*1000,
  - all later slots zero.
The architecture is causal for slot i (TTT updates only on SRC+clean chunks
before it; SWA windows never cross a chunk boundary except noisy->previous
clean), so the prediction at slot i is EXACTLY the conditional the model was
trained on — the ccv conditioning (source written into fast weights via the
update path) is bit-identical to training. The incremental inference kernels
(ar_..._inference) support neither the SRC prefix nor the hidden/input camera
rotary; reusing the training path avoids re-implementing (and re-validating)
all four variants at the cost of extra compute (one full-sequence forward per
step; ~n_steps x 7 forwards per video).

Determinism: per-pair CUDA generator seeded from the pair's position in the
fixed list; no CFG by default (guide_scale 1.0).

Usage (from lact_ar_video/minVid, PYTHONPATH=<repo>/lact_ar_video):
  python eval_ccv_generate.py --config configs/ar/abl_ccv_base.yaml \
      --ckpt ../outputs/ccv_base/seed_1/checkpoint_model_013999 \
      --out ../outputs/eval_dev/gen_ccv_base_13999 --n_pairs 2 [--steps 40]
"""
import argparse
import json
import os
import time

import eval_ccv_common as common  # noqa: F401  (sets env before torch use)
import torch
from einops import rearrange

from eval_ccv_common import (
    DEFAULT_PAIRS_JSON,
    cache_and_free_text_encoder,
    load_config,
    load_model,
    load_or_build_pairs,
    make_pair_dataset,
    pair_batch_to_gpu,
    psnr_per_frame,
    ssim_per_frame,
)
from minVid.models.blocks.cam_phase_builder import build_ccv_cam_inputs
from minVid.utils.io_utils import save_video

GEN_SEED_BASE = 424242  # per-pair generation seed = GEN_SEED_BASE + index


def shifted_sigmas(n_steps, shift, device):
    """sigma(u) = shift*u / (1 + (shift-1)*u), u = 1 -> 0 (training warp)."""
    u = torch.linspace(1.0, 0.0, n_steps + 1, device=device, dtype=torch.float64)
    return (shift * u / (1.0 + (shift - 1.0) * u)).float()


@torch.no_grad()
def generate_target_video(model, batch, n_steps, shift, seed, guide_scale=1.0,
                          uncond_prompt="", device="cuda"):
    """AR-generate the target-view latent video for one ccv pair.

    Returns (tgt_latent_gen [1, F_lat, C, H, W] fp32, timing dict).
    """
    caption = batch["text_prompts"][0]
    fw = model.ar_window_size  # 3 latent frames per chunk

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        src_latent = model.vae.encode(batch["video_rgb_src"] * 2.0 - 1.0)
    src_latent = src_latent.float()  # [1, f_src, c, h, w]
    _, f_src, c_lat, h_lat, w_lat = src_latent.shape
    n_src_chunks = f_src // fw
    n_tgt_f = int(model.src_latent_f) if model.src_latent_f > 0 else f_src
    assert f_src == n_tgt_f, "src/tgt latent frame count mismatch"
    n_chunks = n_tgt_f // fw
    n_slots = n_src_chunks + 2 * n_chunks - 1  # 7 + 13 = 20

    # camera conditioning, built exactly as _forward_pair does
    cam12_arg, coords_arg = None, None
    if model.use_cam_encoder or model.cam_phase_mode == "plucker":
        with torch.autocast(device_type="cuda", enabled=False):
            cam12_per_frame, coords6 = build_ccv_cam_inputs(
                batch["c2w_src"][0].float(),
                batch["c2w_tgt"][0].float(),
                batch["K"][0].float(),
                latent_hw=(h_lat // 2, w_lat // 2),
                n_latent_f=n_tgt_f,
                ar_window_f=fw,
            )
        if model.use_cam_encoder:
            cam12_arg = cam12_per_frame[None].to(device)
        if model.cam_phase_mode == "plucker":
            coords_arg = coords6[None].to(device)

    # text embeddings (cached stub)
    text_embeds = model.text_encoder([caption])["prompt_embeds"]  # [1, L, D]
    text_rep = text_embeds.expand(n_slots, -1, -1)
    if guide_scale != 1.0:
        text_uc = model.text_encoder([uncond_prompt])["prompt_embeds"]
        text_uc_rep = text_uc.expand(n_slots, -1, -1)

    # sequence buffer [n_slots, fw, c, h, w]; SRC prefix = clean source chunks
    ar_input = src_latent.new_zeros(n_slots, fw, c_lat, h_lat, w_lat)
    ar_input[:n_src_chunks] = src_latent.reshape(n_src_chunks, fw, c_lat, h_lat, w_lat)
    ar_t = torch.zeros(n_slots, device=device, dtype=torch.float32)
    ar_seq_len = (fw * h_lat * w_lat) // 4

    sigmas = shifted_sigmas(n_steps, shift, device)
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    def forward_velocity(slot):
        flow_pred, _ = model.generator(
            ar_input.clone(),
            {"prompt_embeds": text_rep},
            ar_t,
            convert_to_x0=False,
            seq_len=ar_seq_len,
            cam12_per_frame=cam12_arg,
            cam_coords6=coords_arg,
        )
        v = flow_pred[slot].float()
        if guide_scale != 1.0:
            flow_uc, _ = model.generator(
                ar_input.clone(),
                {"prompt_embeds": text_uc_rep},
                ar_t,
                convert_to_x0=False,
                seq_len=ar_seq_len,
                cam12_per_frame=cam12_arg,
                cam_coords6=coords_arg,
            )
            v = flow_uc[slot].float() + guide_scale * (v - flow_uc[slot].float())
        return v

    tgt_latent = src_latent.new_zeros(1, n_tgt_f, c_lat, h_lat, w_lat)
    tic = time.time()
    for i in range(n_chunks):
        noisy_slot = n_src_chunks + 2 * i
        x = torch.randn(fw, c_lat, h_lat, w_lat, generator=g, device=device,
                        dtype=torch.float32)  # x at sigma=1 is pure noise
        for s in range(n_steps):
            sig, sig_next = sigmas[s].item(), sigmas[s + 1].item()
            ar_input[noisy_slot] = x
            ar_t.zero_()
            ar_t[noisy_slot] = sig * model.num_train_timestep
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v = forward_velocity(noisy_slot)
            x = x + (sig_next - sig) * v  # Euler on dx/dsigma = velocity
        tgt_latent[0, i * fw : (i + 1) * fw] = x
        if i < n_chunks - 1:
            ar_input[noisy_slot + 1] = x  # clean slot for the next AR steps
        ar_t.zero_()
        print(f"    chunk {i + 1}/{n_chunks} done "
              f"({time.time() - tic:.1f}s cumulative)", flush=True)

    return tgt_latent, {"gen_seconds": time.time() - tic}


@torch.no_grad()
def compute_metrics(gen_frames, gt_frames, lpips_model, device):
    """gen/gt: [F, C, H, W] float in [0, 1] on device. Returns dict."""
    psnr = psnr_per_frame(gen_frames, gt_frames)
    ssim = ssim_per_frame(gen_frames, gt_frames)
    lp = []
    for s in range(0, gen_frames.shape[0], 16):
        a = gen_frames[s : s + 16].float() * 2.0 - 1.0
        b = gt_frames[s : s + 16].float() * 2.0 - 1.0
        lp.append(lpips_model(a, b).flatten())
    lp = torch.cat(lp)
    return {
        "psnr_mean": psnr.mean().item(),
        "ssim_mean": ssim.mean().item(),
        "lpips_mean": lp.mean().item(),
        "psnr_per_frame": [round(v, 4) for v in psnr.tolist()],
        "ssim_per_frame": [round(v, 5) for v in ssim.tolist()],
        "lpips_per_frame": [round(v, 5) for v in lp.tolist()],
    }


def to_uint8_video(frames):
    """[F, C, H, W] in [0,1] -> uint8 [F, C, H, W] on cpu."""
    return (frames.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


def main():
    parser = argparse.ArgumentParser(description="ccv Phase-2 generation eval")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--pairs", type=str, default=DEFAULT_PAIRS_JSON)
    parser.add_argument("--out", type=str, required=True,
                        help="output DIR (mp4s + metrics json)")
    parser.add_argument("--n_pairs", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=40,
                        help="Euler denoising steps per chunk")
    parser.add_argument("--shift", type=float, default=-1.0,
                        help="sigma-schedule shift; -1 = training shift "
                             "(adjust_timestep_shift value)")
    parser.add_argument("--guide_scale", type=float, default=1.0,
                        help="text CFG scale; 1.0 = off (default)")
    parser.add_argument("--uncond_prompt", type=str, default="")
    parser.add_argument("--save_videos", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    config = load_config(args.config)
    payload = load_or_build_pairs(args.pairs, config)
    pairs = payload["pairs"]
    if args.n_pairs > 0:
        pairs = pairs[: args.n_pairs]

    model = load_model(config, args.ckpt, device=args.device)
    caption = config.dataset_train.params.caption
    prompts = [caption] + ([args.uncond_prompt] if args.guide_scale != 1.0 else [])
    model = cache_and_free_text_encoder(model, prompts, device=args.device)
    model.eval()

    shift = args.shift if args.shift > 0 else float(model.timestep_shift)
    print(f"[generate] sampler: Euler, {args.steps} steps, shift {shift:.4f}, "
          f"guide_scale {args.guide_scale}")

    import lpips

    lpips_model = lpips.LPIPS(net="alex").to(args.device).eval()

    ds = make_pair_dataset(config, pairs)
    os.makedirs(args.out, exist_ok=True)

    results = []
    for i in range(len(pairs)):
        rec = {
            "index": i,
            "relpath": pairs[i]["relpath"],
            "src_cam": pairs[i]["src_cam"],
            "tgt_cam": pairs[i]["tgt_cam"],
            "seed": GEN_SEED_BASE + i,
        }
        print(f"[generate] pair {i}/{len(pairs)}: {rec['relpath']} "
              f"cams({rec['src_cam']},{rec['tgt_cam']})", flush=True)
        item = ds[i]
        batch = pair_batch_to_gpu(item, args.device)

        tgt_latent, timing = generate_target_video(
            model, batch, n_steps=args.steps, shift=shift,
            seed=GEN_SEED_BASE + i, guide_scale=args.guide_scale,
            uncond_prompt=args.uncond_prompt, device=args.device,
        )

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_px = model.vae.decode_to_pixel(tgt_latent)  # [1, C, F, H, W] in [-1,1]
        gen_frames = (gen_px[0].float().permute(1, 0, 2, 3) + 1.0) / 2.0  # [F,C,H,W]
        gt_frames = batch["video_rgb_tgt"][0].float()  # [F, C, H, W] in [0,1]

        # decoded frame count can differ by VAE conv padding; align defensively
        n_f = min(gen_frames.shape[0], gt_frames.shape[0])
        gen_frames, gt_frames = gen_frames[:n_f], gt_frames[:n_f]

        rec.update(compute_metrics(gen_frames, gt_frames, lpips_model, args.device))
        rec.update(timing)
        results.append(rec)
        print(f"[generate]   PSNR {rec['psnr_mean']:.3f} dB  "
              f"SSIM {rec['ssim_mean']:.4f}  LPIPS {rec['lpips_mean']:.4f}  "
              f"({rec['gen_seconds']:.0f}s)", flush=True)

        if args.save_videos:
            tag = (f"pair{i:03d}_{rec['relpath'].replace(os.sep, '_')}"
                   f"_cam{rec['src_cam']:02d}to{rec['tgt_cam']:02d}")
            gen_u8 = to_uint8_video(gen_frames)
            gt_u8 = to_uint8_video(gt_frames)
            src_u8 = to_uint8_video(batch["video_rgb_src"][0].float()[:n_f])
            save_video(gen_u8, os.path.join(args.out, f"{tag}_gen.mp4"), save_fps=15)
            # side-by-side [src | gen | gt] for eyeballing
            sxs = torch.cat([src_u8, gen_u8, gt_u8], dim=3)
            save_video(sxs, os.path.join(args.out, f"{tag}_src_gen_gt.mp4"), save_fps=15)

    summary = {
        "config": os.path.abspath(args.config),
        "ckpt": os.path.abspath(args.ckpt),
        "pairs_file": os.path.abspath(args.pairs),
        "sampler": {"type": "euler_full_seq_teacher_forcing",
                    "steps": args.steps, "shift": shift,
                    "guide_scale": args.guide_scale,
                    "gen_seed_base": GEN_SEED_BASE},
        "n_pairs": len(results),
        "psnr_mean": sum(r["psnr_mean"] for r in results) / len(results),
        "ssim_mean": sum(r["ssim_mean"] for r in results) / len(results),
        "lpips_mean": sum(r["lpips_mean"] for r in results) / len(results),
        "per_pair": results,
    }
    out_json = os.path.join(args.out, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[generate] SUMMARY over {len(results)} pairs: "
          f"PSNR {summary['psnr_mean']:.3f} dB, SSIM {summary['ssim_mean']:.4f}, "
          f"LPIPS {summary['lpips_mean']:.4f}")
    print(f"[generate] wrote {out_json}")


if __name__ == "__main__":
    main()
