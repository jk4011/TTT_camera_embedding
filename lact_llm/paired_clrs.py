# -*- coding: utf-8 -*-
"""Paired per-problem comparison of Q29 CLRS arms.

The val_log only records a pooled answer_acc over the whole held-out set, which
cannot say whether a gap survives seed/scene noise. Every arm is evaluated on
the SAME held-out blocks in the SAME order, so accuracy can be scored per block
and the arms compared as a paired sample: delta_i = acc_i(A) - acc_i(B) over
blocks i, with a paired t-statistic and a win rate alongside the mean.

Pairing matters here because CLRS blocks differ enormously in difficulty (the
answer region is 3-11% of the sequence and the algorithms vary), so the
between-block variance dwarfs the between-arm effect. Unpaired error bars on
these numbers would be far too wide to resolve a 0.001-0.015 gap.

Usage:
    python paired_clrs.py --gpu 3 --arms q29_base q29_in_2d q29_h_2d ...
"""

import argparse
import json
import math
import os

import torch

import clrs_data
import train_small
from lact_model import LaCTForCausalLM, LaCTSWIGLUConfig


def load_arm(out_dir, device):
    """Rebuild an arm's model from final.pt.

    Only the state_dict is saved, so the config is recovered from the `[cfg]`
    block the trainer prints at startup. That block IS the config actually used
    (including the per-arm extra_json flags such as ttt_hidden_rope), so it is
    the authoritative record -- reconstructing from CLI defaults instead would
    silently rebuild the wrong arm.
    """
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
    sd = torch.load(os.path.join(out_dir, "final.pt"), map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def per_block_acc(model, val_set, coord_mode, device, bs=8):
    """Answer-token accuracy for EACH block separately -> [n_blocks] float."""
    ttt_layers = [blk.attn for blk in model.model.layers]
    out = []
    for i in range(0, val_set.shape[0], bs):
        blk = val_set[i:i + bs].to(device)
        tokens, coords, labels = train_small.clrs_split(blk, coord_mode)
        train_small.set_ext_coords(ttt_layers, coords)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=tokens).logits
        pred = logits[:, :-1, :].float().argmax(-1)
        tgt = labels[:, 1:]
        sel = tgt != -100
        hit = ((pred == tgt) & sel).sum(dim=1).float()
        tot = sel.sum(dim=1).float()
        # a block with no supervised token carries no information; drop it
        out.extend((hit[j] / tot[j]).item() if tot[j] > 0 else float("nan")
                   for j in range(blk.shape[0]))
    train_small.set_ext_coords(ttt_layers, None)
    return out


def paired(a, b):
    """Paired stats for a - b over blocks where both are defined."""
    d = [x - y for x, y in zip(a, b)
         if not (math.isnan(x) or math.isnan(y))]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    t = mean / se if se > 0 else float("inf")
    wins = sum(1 for x in d if x > 0)
    ties = sum(1 for x in d if x == 0)
    return dict(n=n, mean=mean, se=se, t=t,
                win_rate=wins / n, tie_rate=ties / n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--quadrant", type=str, default="2d_long")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--val_tokens", type=int, default=2000000)
    p.add_argument("--arms", nargs="+", required=True,
                   help="output dir names under outputs/")
    p.add_argument("--out", type=str, default="outputs/q29_paired.json")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    n_val_blocks = max(1, args.val_tokens // args.seq_len)
    val_set = clrs_data.get_or_build_clrs_val_set(
        n_val_blocks, clrs_data.val_cache_path(args.quadrant, args.seq_len),
        args.quadrant, args.seq_len)
    print(f"[paired] val set {tuple(val_set.shape)}", flush=True)

    # coord_mode is a property of the arm, recorded in its own train.log
    scores = {}
    for name in args.arms:
        out_dir = os.path.join("outputs", name)
        mode = "2d" if name.endswith("_2d") else "1d"
        model = load_arm(out_dir, device)
        scores[name] = per_block_acc(model, val_set, mode, device)
        del model
        torch.cuda.empty_cache()
        ok = [s for s in scores[name] if not math.isnan(s)]
        print(f"[paired] {name:<14} mode={mode} pooled={sum(ok)/len(ok):.4f}",
              flush=True)

    results = {"pooled": {k: sum(s for s in v if not math.isnan(s))
                          / sum(1 for s in v if not math.isnan(s))
                          for k, v in scores.items()},
               "pairs": {}}
    base = args.arms[0]
    for name in args.arms[1:]:
        results["pairs"][f"{name}_vs_{base}"] = paired(scores[name], scores[base])
    # the load-bearing within-dimension contrasts
    for a, b in (("q29_h_2d", "q29_in_2d"), ("q29_h_1d", "q29_in_1d"),
                 ("q29_in_2d", "q29_in_1d"), ("q29_h_2d", "q29_h_1d")):
        if a in scores and b in scores:
            results["pairs"][f"{a}_vs_{b}"] = paired(scores[a], scores[b])

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'contrast':<28} {'mean d':>9} {'t':>8} {'win':>7}  n")
    for k, v in results["pairs"].items():
        print(f"{k:<28} {v['mean']:>+9.4f} {v['t']:>8.2f} "
              f"{v['win_rate']:>6.1%}  {v['n']}")
    print(f"\n[paired] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
