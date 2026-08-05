# NODE2 (2026-08-05): restart the CCV site ablation with one line changed

`git pull` first, then paste this whole file into the node2 session. This supersedes
the CCV section of `NODE2_PROMPT.md`.

---

## Stop `ccv_site_in`, `ccv_site_h`, `ccv_site_both` now

You launched them at commit `94b4fd6` from the arm definition that was in the prompt at
the time. That definition was incomplete. Each run is ~46 h, so the sooner they stop the
less is wasted.

**What is wrong:** `ttt_hrope_frac` was left at its default `0.5`, so the hidden site
rotates only **50%** of the hidden dimensions while the input site rotates 98.4%. Every
task CCV is compared against sits differently:

| task | fast q/k rotated | hidden rotated |
|---|---|---|
| NVS | 98.4% | 98.4% |
| 3D reconstruction (original) | 25.0% | **8.2%** |
| 3D reconstruction (re-running on node1 now) | 100% | 100% |
| CCV as you launched it | 98.4% | **50%** |

The 8.2% setting is the one place where `Both` loses to the single-site arms, and it
loses more as input views grow: at 16 views `Both` is behind `input` by 0.40 dB and is
better on only 9 of 140 scenes. That is why node1 is re-running 3D reconstruction at
100/100. If CCV stays at 50% we cannot tell whether an input-vs-hidden difference there
is about the SITE or about the LADDER WIDTH, which is the whole question the ablation
exists to answer.

**Kill order matters.** Retry/self-heal parents first, or they relaunch the trainers:

1. retry parents,
2. then the trainers,
3. then the locks `lact_nvs/outputs/.gpu_locks/node2_gpu{0,1,2}`,
4. then the queue entries, so a resubmission does not restart them.

Do **not** use `pkill -f`, and do not use `ps | grep <pattern>` either: both match your
own command line. node1 killed its own shell that way earlier today. Kill by PID,
excluding `$$`, after printing exactly what you are about to kill.

Keep the partial checkpoints; do not delete `outputs/ccv_site_*`.

## Relaunch the same three cells, one line different

From `abl_ccv_both.yaml` (it already has `use_cam_encoder: true`):

```yaml
use_cam_encoder: true          # the headline pair, ccv_base vs ccv_both, both have it
cam_phase_mode: plucker
ttt_learnable_freqs: false     # fixed ladder for headlines
ttt_hrope_frac: 1.0            # <-- THE FIX: 100% of hidden dims, not the 0.5 default
ttt_input_rope:  true / false  # input: true,  hidden: false, both: true
ttt_hidden_rope: false / true  # input: false, hidden: true,  both: true
```

Already checked, so you need not derive it: at `frac: 1.0` the hidden ladder is
`h_rope_dim = 1536/1536` with `nf_h = 128`, and the assert `2*P_h <= d_h` passes exactly.
The input site cannot reach a literal 100%, because `nf_in = (768-12)/12 = 63`
deliberately leaves at least 12 dimensions unrotated; 98.4% is the same as NVS.

`ccv_base` stays valid and needs no re-run: it has no rotary, so the fraction does not
apply. It is the baseline row for all three new cells.

## Check three things before leaving them alone

1. Each cell's log shows the flags you intended. A cam checkpoint strict-loads into a
   stock model with **zero missing keys**, so a mis-set flag produces a plausible NULL
   with nothing in any log to indicate a problem. Require a positive confirmation.
2. A checkpoint directory actually appears after the first `save_every` interval
   (now 250 steps, about 43 min at the measured 10.3 s/step). A run that silently never
   checkpoints looks identical to a healthy one until Slurm kills it.
3. On any restart the step number continues rather than restarting at 0.

`keep_last_iter` is 1000, so only the 4 most recent checkpoints survive. If you want a
specific step for an eval ladder, copy it aside when it appears.

## Everything else is on hold

Group A (seed completion) is deferred by the user until a node1 experiment reports:
splitting the rotary budget half camera / half 2-D image coordinate. F34 found PRoPE's
entire gain in this stack came from its image-coordinate ropes (+0.379) while its
projective transform cost -0.294, and our ladder currently spends 100% on camera and
nothing on image position. Do not start Group A yet.

If a GPU is free once the three CCV cells are running, say so rather than picking
something up.

## House rules

- **Report paired**, never unpaired: per-pair deltas with a t-statistic, seed-matched,
  win-rate alongside the mean.
- **PAPER FREEZE**: results go to `RESULTS_DOSSIER.md` / `lact_llm/ga_honly/LEDGER.md`.
  Do not edit any `.tex`; node1 maintains `experiment.tex`.
- GPU locks are shared across nodes on lustre: keep the `node2_` prefix.
- Launch as your own background task so the harness notifies you; nohup'd daemons finish
  silently.
