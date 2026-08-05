# -*- coding: utf-8 -*-
"""Paired per-block LM evaluation of the Q31 attn_nope arms.

Two things the val_log cannot give:

1. PAIRED STATS. It records one pooled loss per run, which cannot say whether a
   0.01-0.02 nat gap survives noise. Every arm is scored on the SAME val blocks
   in the SAME order, so the arms compare as a paired sample over blocks.

2. A CLEAN CROSS-COLUMN COMPARISON. F27 (the attn-rope-ON column) was evaluated
   on `val_cache_fla-hub_transformer-1.3B-100B_4096.pt`, which was later deleted
   and replaced by the `_ds42` cache that Q31 used. Absolute ppl across the two
   columns is therefore NOT comparable as logged. Passing --val_cache restores
   the F27-era sample so the Q31 checkpoints can be re-scored on it, which makes
   the ON/OFF comparison a same-sample comparison without retraining anything.

Usage:
    python paired_lm.py --gpu 0 --val_cache val_cache_...4096.pt \
        --arms q31_attnnope_nope q31_attnnope_in q31_attnnope_h q31_attnnope_both
"""

import argparse
import json
import math
import os

import torch

from lact_model import LaCTForCausalLM, LaCTSWIGLUConfig


def load_arm(out_dir, device):
    """Rebuild an arm from final.pt; config comes from its logged [cfg] block."""
    with open(os.path.join(out_dir, "train.log")) as f:
        log = f.read()
    start = log.index("[cfg] LaCTSWIGLUConfig ") + len("[cfg] LaCTSWIGLUConfig ")
    depth, end = 0, None
    for i in range(start, len(log)):
        if log[i] == "{":
            depth += 1
        elif log[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    cfg = json.loads(log[start:end])
    model = LaCTForCausalLM(LaCTSWIGLUConfig(**cfg)).to(device)
    model.load_state_dict(torch.load(os.path.join(out_dir, "final.pt"),
                                     map_location=device))
    model.eval()
    return model, cfg


@torch.no_grad()
def per_block_loss(model, val_set, device, bs=8):
    """Mean next-token loss for EACH val block separately -> [n_blocks]."""
    out = []
    for i in range(0, val_set.shape[0], bs):
        x = val_set[i:i + bs].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x).logits
        # the model shifts internally: logits at t score the token at t+1
        lp = logits[:, :-1, :].float()
        tgt = x[:, 1:]
        ls = torch.nn.functional.cross_entropy(
            lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction="none")
        out.extend(ls.view(tgt.shape).mean(dim=1).tolist())
    return out


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    return dict(n=n, mean=mean, se=se, t=(mean / se if se > 0 else float("inf")),
                win_rate=sum(1 for x in d if x < 0) / n)  # lower loss = win


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--val_cache", type=str, required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--out", type=str, default="outputs/q31_paired.json")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    val_set = torch.load(args.val_cache, map_location="cpu")
    print(f"[paired-lm] val {tuple(val_set.shape)} from {args.val_cache}",
          flush=True)

    scores = {}
    for name in args.arms:
        model, cfg = load_arm(os.path.join("outputs", name), device)
        # provenance: the arm is only what its own config says it is
        print(f"[paired-lm] {name:<26} attn_nope={cfg.get('attn_nope')} "
              f"ttt_nope={cfg.get('ttt_nope')} "
              f"hidden={cfg.get('ttt_hidden_rope', False)}", flush=True)
        scores[name] = per_block_loss(model, val_set, device)
        del model
        torch.cuda.empty_cache()
        m = sum(scores[name]) / len(scores[name])
        print(f"[paired-lm] {name:<26} loss={m:.4f} ppl={math.exp(m):.4f}",
              flush=True)

    pooled = {k: sum(v) / len(v) for k, v in scores.items()}
    res = {"val_cache": args.val_cache,
           "pooled_loss": pooled,
           "pooled_ppl": {k: math.exp(v) for k, v in pooled.items()},
           "pairs": {}}
    ref = args.arms[0]
    for name in args.arms[1:]:
        res["pairs"][f"{name}_vs_{ref}"] = paired(scores[name], scores[ref])

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n{'contrast':<42} {'mean dloss':>11} {'t':>8} {'win':>7}  n")
    for k, v in res["pairs"].items():
        print(f"{k:<42} {v['mean']:>+11.4f} {v['t']:>8.2f} "
              f"{v['win_rate']:>6.1%}  {v['n']}")
    print(f"\n[paired-lm] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
