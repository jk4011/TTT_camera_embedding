"""Figure 3: splitting the fast-weight update into n sequential chunks.

Replaces the update-count table: all four arms drawn, absolute PSNR on the left and
the paired per-scene delta over NoPE on the right. Data is the multi-chunk grid
(mc_eval/): one model per arm, trained at 32 input views with the chunk count sampled
uniformly from {1,2,4,8}, evaluated at each n on 256 held-out scenes, seed 95.

Colors/markers must stay identical to make_fig1.py: color follows the arm across
every figure in the paper, and marker shape is the colorblind-safe secondary code.
"""
import json, math, os
import statistics as st
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType, not Type 3
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/paper_overleaf/fig3_update_steps.pdf"
STYLE = [("NoPE", "base", "#888888", "o", "--"),
         ("input RoPE", "in", "#1f77b4", "s", "-"),
         ("hidden RoPE", "h", "#2ca02c", "^", "-"),
         ("Both (TTT-RoPE)", "both", "#d62728", "D", "-")]
NS = [1, 2, 4, 8]

D = {}
for lab, cell, *_ in STYLE:
    D[lab] = {n: json.load(open(f"{HERE}/mc_eval/mc_{cell}_n{n}.json"))["per_scene_psnr"]
              for n in NS}
nsc = len(D["NoPE"][1])
assert all(len(v) == nsc for d in D.values() for v in d.values()), "scene sets differ"

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# left: absolute PSNR -- NoPE collapses as the update goes sequential, the arms hold
for lab, cell, c, mk, ls in STYLE:
    ax[0].plot(NS, [st.mean(D[lab][n]) for n in NS], marker=mk, color=c, ls=ls,
               label=lab, lw=1.8, ms=5)
ax[0].set_ylabel("PSNR")
ax[0].set_title("absolute")

# right: paired per-scene delta over NoPE, with paired-stderr error bars
for lab, cell, c, mk, ls in STYLE[1:]:
    mu, se = [], []
    for n in NS:
        d = [a - b for a, b in zip(D[lab][n], D["NoPE"][n])]
        m = st.mean(d)
        mu.append(m)
        se.append(st.stdev(d) / math.sqrt(len(d)))
    ax[1].errorbar(NS, mu, yerr=se, marker=mk, color=c, ls=ls, label=lab,
                   lw=1.8, ms=5, capsize=2)
ax[1].axhline(0, color="k", lw=.7, alpha=.4)
ax[1].set_ylabel(r"$\Delta$ PSNR vs NoPE (paired)")
ax[1].set_title("paired delta over NoPE")

for a in ax:
    a.set_xscale("log", base=2)
    a.set_xticks(NS); a.set_xticklabels(NS)
    a.set_xlabel("sequential update chunks $n$")
    a.legend(fontsize=8, loc="best"); a.grid(alpha=.25)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}  ({nsc} scenes/point)")
