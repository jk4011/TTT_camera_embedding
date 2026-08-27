"""video2 generation eval: teacher-forced continuation (PSNR/SSIM/LPIPS) and
free-running rollout (mp4s for FVD).

Sampler: same full-sequence design as eval_ccv_generate.py -- every Euler
denoise step runs the TRAINING forward on the full 13-slot interleave
[noisy0, clean0, ..., noisy6], with
  - clean slot j < i : GT latents (--mode tf) or previously generated chunks
                       (--mode fr),
  - noisy slot i     : current x_sigma at timestep sigma*1000,
  - all later slots  : zero.
Zero slots contribute exactly zero gradient to the t5 gathered update (k=v=0
=> e=0), so the prediction at slot i is conditioned only on chunks < i: no
leakage of the chunk being generated or of future chunks. Clean slots are
filled only AFTER their chunk is finished (never prefilled), which is what
enforces this for the single-gathered-update kernel.

Teacher forcing (--mode tf): clean slot i receives the GT latent chunk, so
every generated chunk i >= 1 is "the model's next 3 latent frames given the
real past" and has an exact GT counterpart -> per-frame PSNR/SSIM/LPIPS are
meaningful (chunk 0 is unconditional; exclude its pixel frames [0:9) in
analysis). Free running (--mode fr): clean slot receives the generated chunk;
mp4s feed the FVD computation.

Determinism: one CUDA generator per clip seeded by its GLOBAL index, so the
per-chunk noise draws are identical across arms -> fully paired.

Usage (from lact_ar_video/minVid, PYTHONPATH=<repo>/lact_ar_video):
  python eval_video2_generate.py --config configs/ar/video2_t5_base.yaml \
      --ckpt ../outputs/video2_t5_base/seed_1/checkpoint_model_001499 \
      --out ../outputs/eval_dev/gen_video2_t5_base --mode tf --n_clips 24
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
    psnr_per_frame,
    ssim_per_frame,
)
from minVid.utils.io_utils import save_video

GEN_SEED_BASE = 424242  # per-clip generation seed = GEN_SEED_BASE + index


def shifted_sigmas(n_steps, shift, device):
    """sigma(u) = shift*u / (1 + (shift-1)*u), u = 1 -> 0 (training warp)."""
    u = torch.linspace(1.0, 0.0, n_steps + 1, device=device, dtype=torch.float64)
    return (shift * u / (1.0 + (shift - 1.0) * u)).float()


@torch.no_grad()
def generate_video(model, gt_latent, text_embeds, n_steps, shift, seed,
                   teacher_forced, device="cuda"):
    """AR-generate one 21-latent-frame video. gt_latent: [1, 21, c, h, w] fp32.

    Returns (gen_latent [1, 21, c, h, w] fp32, seconds).
    """
    fw = model.ar_window_size  # 3
    _, n_f, c_lat, h_lat, w_lat = gt_latent.shape
    n_chunks = n_f // fw
    n_slots = 2 * n_chunks - 1

    ar_input = gt_latent.new_zeros(n_slots, fw, c_lat, h_lat, w_lat)
    ar_t = torch.zeros(n_slots, device=device, dtype=torch.float32)
    ar_seq_len = (fw * h_lat * w_lat) // 4
    text_rep = text_embeds.expand(n_slots, -1, -1)

    sigmas = shifted_sigmas(n_steps, shift, device)
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    gen_latent = gt_latent.new_zeros(1, n_f, c_lat, h_lat, w_lat)
    tic = time.time()
    for i in range(n_chunks):
        noisy_slot = 2 * i
        x = torch.randn(fw, c_lat, h_lat, w_lat, generator=g, device=device,
                        dtype=torch.float32)
        for s in range(n_steps):
            sig, sig_next = sigmas[s].item(), sigmas[s + 1].item()
            ar_input[noisy_slot] = x
            ar_t.zero_()
            ar_t[noisy_slot] = sig * model.num_train_timestep
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                flow_pred, _ = model.generator(
                    ar_input.clone(),
                    {"prompt_embeds": text_rep},
                    ar_t,
                    convert_to_x0=False,
                    seq_len=ar_seq_len,
                )
            v = flow_pred[noisy_slot].float()
            x = x + (sig_next - sig) * v  # Euler on dx/dsigma = velocity
        gen_latent[0, i * fw : (i + 1) * fw] = x
        if i < n_chunks - 1:
            if teacher_forced:
                ar_input[noisy_slot + 1] = gt_latent[0, i * fw : (i + 1) * fw]
            else:
                ar_input[noisy_slot + 1] = x
        ar_t.zero_()
    return gen_latent, time.time() - tic


@torch.no_grad()
def compute_metrics(gen_frames, gt_frames, lpips_model):
    """gen/gt: [F, C, H, W] float in [0, 1] on device."""
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


def main():
    parser = argparse.ArgumentParser(description="video2 generation eval")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True, help="output DIR")
    parser.add_argument("--mode", type=str, choices=["tf", "fr"], default="tf")
    parser.add_argument("--n_clips", type=int, default=24)
    parser.add_argument("--start", type=int, default=0,
                        help="global index of the first clip (shard offset); "
                             "seeds and record ids stay global")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--shift", type=float, default=-1.0)
    parser.add_argument("--save_videos", type=int, default=1)
    parser.add_argument("--save_sxs", type=int, default=4,
                        help="save side-by-side [gen | gt] mp4 for this many clips")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    cfg = load_config(args.config)
    model = load_model(cfg, args.ckpt, args.device)
    dp = cfg.dataset_train.params

    from eval_video2_valloss import build_heldout
    n_total = args.start + args.n_clips
    dataset, names, caps = build_heldout(cfg, n_total)
    model = cache_and_free_text_encoder(model, sorted(set(caps)), args.device)
    model.eval()

    shift = args.shift if args.shift > 0 else float(model.timestep_shift)
    print(f"[gen2] sampler: Euler, {args.steps} steps, shift {shift:.4f}, "
          f"mode {args.mode}")

    import lpips
    lpips_model = lpips.LPIPS(net="alex").to(args.device).eval()

    os.makedirs(args.out, exist_ok=True)
    partial_path = os.path.join(args.out,
                                f"metrics_partial_{args.mode}_{args.start:03d}.json")
    results = []
    if os.path.isfile(partial_path):
        results = json.load(open(partial_path))["per_clip"]
        print(f"[gen2] resuming: {len(results)} clips done", flush=True)
    done = {r["index"] for r in results}

    for local_i in range(args.n_clips):
        i = args.start + local_i
        if i in done:
            continue
        item = dataset[i]
        gt_frames = item["frames"].permute(1, 0, 2, 3).to(args.device)  # [F,C,H,W]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gt_latent = model.vae.encode(
                gt_frames[None].mul(2.0).sub(1.0)).float()  # [1, 21, c, h, w]
        text_embeds = model.text_encoder([caps[i]])["prompt_embeds"]

        gen_latent, secs = generate_video(
            model, gt_latent, text_embeds, args.steps, shift,
            GEN_SEED_BASE + i, teacher_forced=(args.mode == "tf"),
            device=args.device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_px = model.vae.decode_to_pixel(gen_latent)  # [1,C,F,H,W] [-1,1]
        gen_frames = (gen_px[0].float().permute(1, 0, 2, 3) + 1.0) / 2.0
        n_f = min(gen_frames.shape[0], gt_frames.shape[0])
        gen_frames, gt_frames = gen_frames[:n_f], gt_frames[:n_f]

        rec = {"index": i, "clip": names[i],
               "seed": GEN_SEED_BASE + i, "gen_seconds": secs}
        rec.update(compute_metrics(gen_frames, gt_frames, lpips_model))
        results.append(rec)
        print(f"[gen2] clip {i}: PSNR {rec['psnr_mean']:.3f}  "
              f"SSIM {rec['ssim_mean']:.4f}  LPIPS {rec['lpips_mean']:.4f}  "
              f"({secs:.0f}s)", flush=True)
        results.sort(key=lambda r: r["index"])
        with open(partial_path, "w") as f:
            json.dump({"n_clips": len(results), "per_clip": results}, f, indent=1)

        if args.save_videos:
            u8 = (gen_frames.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
            save_video(u8, os.path.join(args.out, f"clip{i:03d}_gen.mp4"),
                       save_fps=15)
            if local_i < args.save_sxs:
                gt_u8 = (gt_frames.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                save_video(torch.cat([u8, gt_u8], dim=3),
                           os.path.join(args.out, f"clip{i:03d}_gen_gt.mp4"),
                           save_fps=15)

    summary = {
        "config": os.path.abspath(args.config),
        "ckpt": os.path.abspath(args.ckpt),
        "mode": args.mode,
        "sampler": {"type": "euler_full_seq", "steps": args.steps,
                    "shift": shift, "gen_seed_base": GEN_SEED_BASE},
        "n_clips": len(results),
        "psnr_mean": sum(r["psnr_mean"] for r in results) / len(results),
        "ssim_mean": sum(r["ssim_mean"] for r in results) / len(results),
        "lpips_mean": sum(r["lpips_mean"] for r in results) / len(results),
        "per_clip": results,
    }
    out_json = os.path.join(args.out, f"metrics_{args.mode}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"DONE {args.mode}: PSNR {summary['psnr_mean']:.3f}  "
          f"SSIM {summary['ssim_mean']:.4f}  LPIPS {summary['lpips_mean']:.4f} "
          f"-> {out_json}")


if __name__ == "__main__":
    main()
