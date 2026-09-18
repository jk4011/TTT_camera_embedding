"""Pick the scenes for the qualitative figure (paper fig:comparison_nvs, 2026-09-19).

Ranks held-out scenes by how much the winner beats the baselines, using the per-scene
PSNR already stored in each eval.json (no rendering needed). Scene indices are positions
in the eval dataset order, which render_compare.py reproduces.

  python pick_scenes.py --win outputs/re10k_dpchan_pdir_vo_s137/eval.json \
      --lose outputs/base_s137_re/eval.json outputs/re10k_prope_s137/eval.json ... [--top 12]

Prints, for the top scenes by min(win - lose) (i.e. the winner beats EVERY baseline by at
least that much), the per-method PSNR so a visually clear row can be chosen by eye.
"""
import argparse
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--win", required=True, help="eval.json of the method that should look best")
p.add_argument("--lose", nargs="+", required=True, help="eval.json of every other method")
p.add_argument("--top", type=int, default=12)
p.add_argument("--min_psnr", type=float, default=0.0,
               help="drop scenes where the winner is below this (avoid rows that all look broken)")
args = p.parse_args()


def load(p_):
    j = json.load(open(p_))
    return np.asarray(j["per_scene_psnr"], dtype=np.float64), os.path.basename(os.path.dirname(p_))


win, win_name = load(args.win)
others = [load(p_) for p_ in args.lose]
n = min([len(win)] + [len(o) for o, _ in others])
win = win[:n]
gaps = np.stack([win - o[:n] for o, _ in others])          # [M, n]
score = gaps.min(0)                                        # beats every baseline by >= score
ok = win >= args.min_psnr
order = np.argsort(-np.where(ok, score, -1e9))[: args.top]

names = [nm for _, nm in others]
print(f"winner: {win_name}   baselines: {', '.join(names)}   (n={n} scenes)")
head = f"{'scene':>6} {'min gap':>8} {win_name[:14]:>14} " + " ".join(f"{nm[:14]:>14}" for nm in names)
print(head)
for i in order:
    print(f"{i:6d} {score[i]:8.2f} {win[i]:14.2f} "
          + " ".join(f"{o[i]:14.2f}" for o, _ in others))
print("\ncomma list for render_compare.py --scenes:")
print(",".join(str(int(i)) for i in order))
