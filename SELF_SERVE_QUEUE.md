# Self-serve queue: any node with a free GPU takes the next unclaimed item

Every item here fills a cell in `paper_overleaf/experiment.tex` that is currently
empty, or removes a caveat that would otherwise have to be footnoted. Nothing here is
speculative.

## How to claim an item (do this, or two nodes will run the same thing)

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope
git pull
mkdir -p .claims
# atomic: only one node can create the file
if ( set -o noclobber; echo "$(hostname -s) gpu<N> $(date -Is)" > .claims/<ITEM_ID> ) 2>/dev/null; then
  echo "claimed <ITEM_ID>"
  git add .claims/<ITEM_ID> && git commit -q -m "claim <ITEM_ID>" && git push
else
  echo "already claimed by: $(cat .claims/<ITEM_ID>)"
fi
```

When it finishes, append the number to `RESULTS_DOSSIER.md`, delete the claim file,
commit and push. If a node dies mid-run, delete its stale claim so someone else can
pick the item up.

---

## GROUP A: seed completion (highest value, cheapest)

These exist at one seed. The paper currently has to say so in a caption, and Table 1
has to be split into two blocks with two different baselines because of it. Finishing
these collapses the blocks and removes the caveats. Each run is the standard NVS
protocol, about 1.6 h on one B200.

Standard protocol, no deviations: LaCT-LVSM `lact_l6_d256_p16`, RE10K 256x256, 8 input
and 8 target views, 30k iterations, bs16, lr 1e-4, LPIPS loss from step 5k. Evaluate on
256 held-out scenes with 8 uniform inputs and 4 midpoint targets.

| ITEM_ID | what | config / cam mode | seed |
|---|---|---|---|
| `A1_gta_s137` | GTA comparator | as `q15_gta_in_s95` | 137 |
| `A2_gta_s211` | GTA comparator | as `q15_gta_in_s95` | 211 |
| `A3_prope_s211` | PRoPE comparator | as `q15_prope_orig_s95` | 211 |
| `A4_fw4l_base_s137` | depth-4, no rotary | as `fw4l_base_s95` | 137 |
| `A5_fw4l_base_s211` | depth-4, no rotary | as `fw4l_base_s95` | 211 |
| `A6_fw4l_rot4_s137` | depth-4, all 4 sites | as `fw4l_rot4_s95` | 137 |
| `A7_fw4l_rot4_s211` | depth-4, all 4 sites | as `fw4l_rot4_s95` | 211 |

PRoPE already has s95 and s137, so it needs only s211. GTA and both depth-4 cells have
s95 only.

**Why this matters more than it looks.** F18 measured the baseline seed spread at
0.35 dB, which is larger than PRoPE's entire reported gain. A single-seed comparator
cannot be compared against a three-seed baseline, which is why Table 1 currently prints
two separate NoPE rows.

---

## GROUP B: missing cells

| ITEM_ID | what | cost |
|---|---|---|
| `B1_ccv_input` | CCV site ablation, input only | ~46 h, 1 GPU |
| `B2_ccv_hidden` | CCV site ablation, hidden only | ~46 h, 1 GPU |
| `B3_ccv_both_fixed` | CCV both sites, fixed ladder, 100% hidden | ~46 h, 1 GPU |
| `B4_video_input` | plain video, input only | ~55 h, 1 GPU |
| `B5_video_both` | plain video, both sites | ~55 h, 1 GPU |

**B1-B3 arm definition** (this was got wrong once already, see F30's label correction).
Start from `abl_ccv_both.yaml`, which already has `use_cam_encoder: true`, and set:

```yaml
use_cam_encoder: true          # the headline pair (ccv_base vs ccv_both) both have it
cam_phase_mode: plucker
ttt_learnable_freqs: false     # fixed ladder for headlines
ttt_hrope_frac: 1.0            # 100% of the hidden dims, NOT the 0.5 default
ttt_input_rope:  true / false  # B1: true,  B2: false, B3: true
ttt_hidden_rope: false / true  # B1: false, B2: true,  B3: true
```

**Why `ttt_hrope_frac: 1.0`.** The existing ccv runs rotate 98.4% of the fast q/k but
only **50%** of the hidden, because `ttt_hrope_frac` defaults to 0.5. Every other task
we compare against sits differently: NVS is 98.4/98.4, and the 3D-reconstruction grid
is being re-run at 100/100 after its original 25/8.2 turned out to be the outlier where
`Both` loses. Holding the hidden width at 50% would leave CCV as the odd one out and
confound any input-vs-hidden comparison with ladder width. At `frac: 1.0` the hidden
ladder is 1536/1536 with `nf_h = 128`; the assert `2*P_h <= d_h` passes exactly.

The input site cannot reach a literal 100%: `nf_in = (768 - 12) / 12 = 63` deliberately
leaves at least 12 dimensions unrotated, giving 98.4%, the same as NVS.

**Consequence for the existing numbers.** `ccv_base` stays valid, since it has no rotary
and the fraction is irrelevant to it. `ccv_pra`, `ccv_pra_fixed` and `ccv_both` are all
50%-hidden runs and are superseded by B1-B3; the CCV column in the paper will be
re-measured except for the baseline row.

**B4/B5 naming trap.** In the plain-video runs `full` means *hidden + learnable
frequencies*, NOT input+hidden, so the existing `full` cell is not the Both arm. Match
F21/F22 exactly (Wan1.3B attention-only finetune, MultiCamVideo, deterministic noise,
20k steps) or the new cells will not be comparable to the existing ones.

---

## GROUP C: the update-count ablation

| ITEM_ID | what |
|---|---|
| `C1_updates` | 1 / 2 / 4 / 8 update steps, NoPE and Both |

**Design it as repeated update steps on ONE chunk, not as more chunks.** F8 already
tested per-view multi-chunk updates and found -0.23 dB, which it attributed to
per-chunk weight-norm decaying earlier views and to 256-token chunks falling below
Muon's amortisation point. F8 explicitly notes that multi-step updates on the same full
chunk were never tested. Implemented as more chunks, this reproduces F8 and answers
nothing.

Can be halved to 4 runs by sweeping only the Both arm.

---

## Rules for whoever runs these

- **Report paired**, never unpaired: per-scene or per-pair deltas with a t-statistic,
  seed-matched, win-rate alongside the mean.
- **PAPER FREEZE**: results go to `RESULTS_DOSSIER.md` /
  `lact_llm/ga_honly/LEDGER.md`. Do not edit any `.tex`; node1 maintains
  `experiment.tex`.
- GPU locks are on lustre and shared between nodes: keep the `<host>_gpu<i>` prefix.
- Launch as your own background task so the harness notifies you; nohup'd daemons
  finish silently.
- Four failure modes we have already hit today, so check for them:
  1. master ports derived from a loop index collide when a launcher is called once per
     cell (`EADDRINUSE`);
  2. kmeans input selection needs n_samples >= n_clusters, so a view sweep cannot go
     below the trained view count;
  3. a log redirect that omits the sweep variable makes per-point checks grep a file
     that does not exist and pass silently;
  4. `ps | grep <pattern>` matches your own command line, exactly like `pkill -f`;
     exclude `$$` or kill by PID only.
