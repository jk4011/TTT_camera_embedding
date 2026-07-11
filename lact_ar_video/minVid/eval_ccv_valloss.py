"""Phase-1 ccv evaluation: paired held-out validation loss.

Computes the EXACT training objective (VideoLatentFlowMatching._forward_pair:
flow-matching MSE with logit-normal weighting over the TGT noisy chunks of the
[SRC clean prefix || TGT AR interleave] sequence) forward-only on a FIXED
held-out pair list, with per-pair deterministic noise and diffusion timesteps
(torch.manual_seed keyed on the pair's position in the fixed list — the same
mechanism as the training runs' deterministic_noise). Two invocations on the
same checkpoint yield identical numbers; different runs/checkpoints evaluated
on the same pair list are paired per (pair, noise, t) for paired-t analysis.

Usage (from lact_ar_video/minVid, PYTHONPATH=<repo>/lact_ar_video):
  python eval_ccv_valloss.py --config configs/ar/abl_ccv_base.yaml \
      --ckpt ../outputs/ccv_base/seed_1/checkpoint_model_013999 \
      --out ../outputs/eval_dev/valloss_ccv_base_13999.json [--n_pairs 4]
"""
import argparse
import json
import os
import time

import eval_ccv_common as common  # noqa: F401  (sets env before torch use)
import torch

from eval_ccv_common import (
    DEFAULT_PAIRS_JSON,
    cache_and_free_text_encoder,
    load_config,
    load_model,
    load_or_build_pairs,
    make_pair_dataset,
    pair_batch_to_gpu,
)

PAIR_SEED_BASE = 777000  # per-pair seed = PAIR_SEED_BASE + pair index


def main():
    parser = argparse.ArgumentParser(description="ccv held-out paired val loss")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint_model_XXXXXX dir (contains dcp/)")
    parser.add_argument("--pairs", type=str, default=DEFAULT_PAIRS_JSON)
    parser.add_argument("--out", type=str, required=True,
                        help="output json path")
    parser.add_argument("--n_pairs", type=int, default=-1,
                        help="evaluate only the first N pairs of the list")
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
    model = cache_and_free_text_encoder(model, [caption], device=args.device)
    model.eval()

    ds = make_pair_dataset(config, pairs)

    results = []
    for i in range(len(pairs)):
        tic = time.time()
        item = ds[i]
        batch = pair_batch_to_gpu(item, args.device)

        # per-pair deterministic noise + timesteps (fixed by list position)
        torch.manual_seed(PAIR_SEED_BASE + i)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)

        rec = {
            "index": i,
            "relpath": pairs[i]["relpath"],
            "src_cam": pairs[i]["src_cam"],
            "tgt_cam": pairs[i]["tgt_cam"],
            "loss": out["loss"].item(),
            "loss_flow": out["loss_flow"].item(),
            "t_chunks": [round(v, 4) for v in out["t"].flatten().tolist()],
        }
        results.append(rec)
        print(f"[valloss] pair {i:3d}/{len(pairs)} {rec['relpath']} "
              f"cams({rec['src_cam']},{rec['tgt_cam']}) "
              f"loss={rec['loss']:.6f} ({time.time() - tic:.1f}s)", flush=True)

    losses = [r["loss"] for r in results]
    summary = {
        "config": os.path.abspath(args.config),
        "ckpt": os.path.abspath(args.ckpt),
        "pairs_file": os.path.abspath(args.pairs),
        "pair_seed_base": PAIR_SEED_BASE,
        "n_pairs": len(results),
        "mean_loss": sum(losses) / len(losses),
        "per_pair": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[valloss] mean loss over {len(results)} pairs: "
          f"{summary['mean_loss']:.6f}")
    print(f"[valloss] wrote {args.out}")


if __name__ == "__main__":
    main()
