"""Q11 Phase-1: deterministic held-out val loss for a recam_ttt checkpoint.

Same protocol shape as the ccv Phase-1 eval: the fixed 64-pair holdout list,
fixed per-pair noise seeds (777000+i), and a small set of FIXED timesteps
(ids 100/500/900 of the 1000-step training schedule) so every variant sees
bit-identical inputs. Reports the unweighted MSE on the target half,
per pair x timestep + means.

Usage (envs/recam, from lact_ar_video):
  PYTHONPATH=.:<ReCamMaster repo> python recam_ttt/eval_valloss.py \
      --config recam_ttt/configs/h.yaml \
      --ckpt outputs/recam_ttt/h/ckpt_step3250.pt \
      --out outputs/recam_ttt/valloss_h_3250.json
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "minVid"))

from recam_ttt.ttt_adapter import set_ctx_phases  # noqa: E402
from recam_ttt.train_recam_ttt import (  # noqa: E402
    HOLDOUT_JSON,
    build_patched_pipe,
    cam_inputs_for_pair,
    load_pair_latents,
    load_prompt_context,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_pairs", type=int, default=64)
    ap.add_argument("--timestep_ids", type=str, default="100,500,900")
    ap.add_argument("--noise_seed_base", type=int, default=777000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = args.device
    tids = [int(t) for t in args.timestep_ids.split(",")]

    pipe, ttt_ctx, _ = build_patched_pipe(cfg, device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    named = dict(pipe.dit.named_parameters())
    for n, t in state["model_trainable"].items():
        named[n].data.copy_(t.to(named[n].dtype))
    print(f"[val] loaded {len(state['model_trainable'])} trainable tensors "
          f"from {args.ckpt} (step {state['step']})", flush=True)
    pipe.dit.eval()

    context = load_prompt_context(cfg.latent_cache, device)
    need_coords = bool(cfg.ttt_input_rope) or bool(cfg.ttt_hidden_rope)
    scheduler = pipe.scheduler

    with open(HOLDOUT_JSON) as f:
        holdout = json.load(f)["pairs"][: args.n_pairs]

    per_pair = []
    with torch.no_grad():
        for i, p in enumerate(holdout):
            rel = p["relpath"]
            src_cam, tgt_cam = int(p["src_cam"]), int(p["tgt_cam"])
            latents = load_pair_latents(cfg.latent_cache, rel, src_cam,
                                        tgt_cam, device)
            cam_emb, coords6 = cam_inputs_for_pair(cfg, rel, src_cam, tgt_cam,
                                                   need_coords, device)
            set_ctx_phases(ttt_ctx, coords6, pipe.dit.blocks[0].ttt_branch)

            g = torch.Generator()
            g.manual_seed(args.noise_seed_base + i)
            noise = torch.randn(latents.shape, generator=g,
                                dtype=torch.float32)
            noise = noise.to(device=device, dtype=latents.dtype)

            tgt_len = latents.shape[2] // 2
            rec = {"index": i, "relpath": rel, "src_cam": src_cam,
                   "tgt_cam": tgt_cam, "losses": {}}
            tic = time.time()
            for tid in tids:
                # 1-D timestep, like the training path
                timestep = scheduler.timesteps[[tid]].to(
                    dtype=torch.bfloat16, device=device)
                noisy = scheduler.add_noise(latents, noise, timestep)
                noisy[:, :, tgt_len:, ...] = latents[:, :, tgt_len:, ...]
                target = scheduler.training_target(latents, noise, timestep)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = pipe.dit(noisy, timestep=timestep, cam_emb=cam_emb,
                                    context=context,
                                    use_gradient_checkpointing=False)
                rec["losses"][str(tid)] = F.mse_loss(
                    pred[:, :, :tgt_len, ...].float(),
                    target[:, :, :tgt_len, ...].float()).item()
            rec["mean_loss"] = sum(rec["losses"].values()) / len(tids)
            rec["seconds"] = time.time() - tic
            per_pair.append(rec)
            print(f"[val] pair {i}/{len(holdout)} {rel} "
                  f"mean {rec['mean_loss']:.6f} ({rec['seconds']:.0f}s)",
                  flush=True)
            _flush(args, cfg, state, tids, per_pair)

    _flush(args, cfg, state, tids, per_pair)
    m = sum(r["mean_loss"] for r in per_pair) / len(per_pair)
    print(f"[val] SUMMARY {cfg.variant}: mean loss {m:.6f} over "
          f"{len(per_pair)} pairs x {len(tids)} timesteps", flush=True)


def _flush(args, cfg, state, tids, per_pair):
    out = {
        "config": os.path.abspath(args.config),
        "variant": str(cfg.variant),
        "ckpt": os.path.abspath(args.ckpt),
        "ckpt_step": int(state["step"]),
        "pairs_file": os.path.abspath(HOLDOUT_JSON),
        "timestep_ids": tids,
        "noise_seed_base": args.noise_seed_base,
        "n_pairs": len(per_pair),
        "mean_loss": (sum(r["mean_loss"] for r in per_pair) / len(per_pair))
        if per_pair else None,
        "mean_loss_per_timestep": {
            str(t): (sum(r["losses"][str(t)] for r in per_pair)
                     / len(per_pair)) if per_pair else None
            for t in tids},
        "per_pair": per_pair,
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, args.out)


if __name__ == "__main__":
    main()
