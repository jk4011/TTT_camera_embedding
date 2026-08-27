"""Twin-leak diagnostic for the t5 single-gathered-update schedule.

For each held-out clip and each chunk k = 1..6, computes the flow-matching MSE
at noisy slot 2k under two memory conditions, on IDENTICAL (clip, chunk,
noise, sigma):
  causal : clean slots j < k filled with GT latents, slots >= k zero
           (exactly the generation-time conditional)
  leaky  : ALL clean slots 0..5 filled with GT latents
           (exactly the training-time conditional, twin included)
Zero clean slots contribute zero gradient to the gathered update, so "causal"
restricts the fast-weight memory to strictly-past chunks.

If the rotary arms' loss advantage comes from phase-addressing the same-
position clean twin (relative phase 0), their advantage should appear in
"leaky" and shrink or reverse in "causal".

Usage: python eval_video2_causal.py --config ... --ckpt ... --out out.json
"""
import argparse
import json
import os
import time

import eval_ccv_common as common  # noqa: F401
import torch

from eval_ccv_common import cache_and_free_text_encoder, load_config, load_model
from eval_video2_valloss import build_heldout

SEED_BASE = 909000
SIGMAS = [0.7, 0.3]  # one high-noise, one low-noise draw per chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n_clips", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = load_config(args.config)
    model = load_model(cfg, args.ckpt, args.device)
    dataset, names, caps = build_heldout(cfg, args.n_clips)
    model = cache_and_free_text_encoder(model, sorted(set(caps)), args.device)
    model.eval()

    fw = model.ar_window_size
    records = []
    t0 = time.time()
    for i in range(args.n_clips):
        item = dataset[i]
        gt_frames = item["frames"].permute(1, 0, 2, 3).to(args.device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gt_latent = model.vae.encode(
                gt_frames[None].mul(2.0).sub(1.0)).float()
        text_embeds = model.text_encoder([caps[i]])["prompt_embeds"]
        _, n_f, c_lat, h_lat, w_lat = gt_latent.shape
        n_chunks = n_f // fw
        n_slots = 2 * n_chunks - 1
        text_rep = text_embeds.expand(n_slots, -1, -1)
        ar_seq_len = (fw * h_lat * w_lat) // 4
        chunks = gt_latent[0].reshape(n_chunks, fw, c_lat, h_lat, w_lat)

        rec = {"clip": names[i]}
        for k in range(1, n_chunks):
            for d, sig in enumerate(SIGMAS):
                g = torch.Generator(device=args.device)
                g.manual_seed(SEED_BASE + i * 1000 + k * 10 + d)
                eps = torch.randn(chunks[k].shape, generator=g,
                                  device=args.device, dtype=torch.float32)
                x_sig = (1.0 - sig) * chunks[k] + sig * eps
                v_tgt = eps - chunks[k]
                for variant in ("causal", "leaky"):
                    ar_input = gt_latent.new_zeros(
                        n_slots, fw, c_lat, h_lat, w_lat)
                    n_fill = k if variant == "causal" else n_chunks - 1
                    for j in range(n_fill):
                        ar_input[2 * j + 1] = chunks[j]
                    ar_input[2 * k] = x_sig
                    ar_t = torch.zeros(n_slots, device=args.device)
                    ar_t[2 * k] = sig * model.num_train_timestep
                    with torch.no_grad(), torch.amp.autocast(
                            "cuda", dtype=torch.bfloat16):
                        flow_pred, _ = model.generator(
                            ar_input, {"prompt_embeds": text_rep}, ar_t,
                            convert_to_x0=False, seq_len=ar_seq_len)
                    mse = (flow_pred[2 * k].float() - v_tgt).pow(2).mean()
                    rec[f"{variant}_k{k}_s{sig}"] = float(mse)
        records.append(rec)
        cm = sum(v for kk, v in rec.items() if kk.startswith("causal")) / 12
        lm = sum(v for kk, v in rec.items() if kk.startswith("leaky")) / 12
        print(f"clip {i}: causal {cm:.5f}  leaky {lm:.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        json.dump({"records": records}, open(args.out, "w"), indent=1)
    print(f"CAUSAL_DONE {args.out}")


if __name__ == "__main__":
    main()
