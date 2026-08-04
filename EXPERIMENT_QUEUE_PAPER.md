# Paper-table queue (everything `paper_overleaf/experiment.md` still shows blank)

One entry per blank cell. Ordered by (value to the paper) / (GPU cost). Nothing here
is speculative work: each item fills a cell that a reader will look for.

Costs assume one B200. Small NVS runs are ~1.6 h each (L6/d256/p16, 30k iters).

---

## RUNNING NOW

### P0. Figure 1, NVS panel: PSNR vs input views [node1, gpu0, eval-only]
`lact_nvs/run_fig1_viewsweep.sh 0` : 3 arms x {2,4,8,16,24,32} views x 256 scenes.
Evaluation only, reuses the 30k/seed-95 checkpoints, co-resides with training.
Writes per-scene arrays, so the panel can carry paired error bars.

**Blocked on P1 for its fourth curve.** No SwiGLU NoPE checkpoint survives on this
node (`outputs/` has `mlp2_base` and `fw4l_base`, both different architectures).

---

## QUEUE

### P1. NVS NoPE baseline, seed 95 [1 run, ~1.6 h] -> ASSIGNED node2 (2026-08-05)
The missing fourth curve of Figure 1, and the row every Table-1 delta is measured
against. Highest value per GPU-hour of anything here.
Config: the stock `lact_l6_d256_p16` recipe, no `cam.mode`. Standard protocol
(RE10K 256x256, 8+8 views, 30k iters, bs16, lr1e-4, LPIPS from 5k).

### P2. Q31 attn_nope, 4 cells [node3, ~2.5 h each]
Already wired, verified, and written into `NODE3_PROMPT.md` as Job 3. Tests the one
remaining alternative explanation for the LLM null: that attention's own rotary
already supplies local position, leaving the fast-weight rotary nothing to add.
Only the attn-rope-OFF column runs; the ON column is F27.

### P3. Video, input-only and Both [2 runs] -> ASSIGNED node2 (2026-08-05)
Table 6 currently has neither. **Naming trap:** in the plain-video runs `full` means
*hidden + learnable frequencies*, not input+hidden, so the existing `full` cell is
NOT the Both arm. Two runs, not one.

### P4. Update-count ablation, 1/2/4/8 [8 runs, ~13 h]
**Design carefully or it answers the wrong question.** F8 already tested *per-view
multi-chunk* updates (more, smaller chunks) and found -0.23 dB, attributed to
per-chunk weight-norm decaying earlier views and to 256-token chunks falling below
Muon's amortisation point. F8 explicitly notes that multi-*step* updates on the SAME
full chunk were never tested. Implement it as repeated update steps on one chunk. If
it is implemented as more chunks it will reproduce F8 and tell us nothing new.
Cost can be halved to 4 runs if only the Both arm is swept (NoPE at each update count
is the more expensive half and arguably not needed).

### P5. CCV hidden-only [1 run] -> ASSIGNED node2 (2026-08-05)
Deferred earlier, now queued. F30 ran base / pra(learnable) / pra_fixed / both, so the
hidden-only cell is the one hole in an otherwise complete table. CCV is the only
video-domain setting where the hidden site earns (+3.9-4.6% over input), so the paper
should be able to show hidden-alone there rather than inferring it.

### P6. GTA and PRoPE, seeds 137 + 211 [4 runs, ~6.4 h]
Decision was that single-seed comparators are acceptable with a caption footnote, so
this is optional. It buys removal of the footnote: our arms are 3-seed against
baseline 21.745, the comparators are single-seed against 21.970, and F18 puts baseline
seed spread at 0.35 dB, larger than PRoPE's entire gain.

### P7. CaPE port + run [implementation + 1 run]
Deferred earlier, now queued at the bottom. Not implemented at all; PRoPE and GTA are.
Only worth it if we expect a reviewer to demand the third comparator.

---

## Node assignment (2026-08-05)

| node | now |
|---|---|
| node1 | tttLRM from-scratch 4 arms (8 GPUs) + Figure 1 NVS sweep (gpu0, eval-only) |
| node2 | **music WAVE 3 CANCELLED**; P1 -> P3 -> P5. See `NODE2_PROMPT.md` |
| node3 | Q29 (3 cells) -> Q31 attn_nope (4 cells) -> Q30 (15 cells). See `NODE3_PROMPT.md` |

Music was cancelled because the REMI 4-cell grid is already complete and is a total
null (spread 0.0069 ppl), its cause is identified (REMI emits Bar/Position tokens, so
position is readable as content), and the user declined the TSD follow-up twice. WAVE 3
fired on its "4 GPUs free" gate before that decision reached the node2 queue.

## Also outstanding, tracked elsewhere

* **tttLRM from-scratch, 4 arms** [node1, 8 GPUs] : Table 2's real row. Ladder at
  5k/10k/15k, then extended to 30k. See `EXPERIMENT_QUEUE.md` Q28.
* **Q29 CLRS-Text 2-D address, 3 cells** and **Q30 N-dimensional tensor recall**
  [node3] : not paper-table work, they test the dimensionality thesis directly.
* **4-layer MLP seed 137** : already running; finishes Table 3's error bar.
* **NoPE w128 seeds 137/211** : completes the 6-cell x 3-seed LLM matrix, whose only
  robust effect is currently the NoPE gap at a single seed.

## Figure 1, CCV panel: x-axis = number of frames (decided 2026-08-05)

Three panels: NVS and 3D reconstruction sweep input views, CCV sweeps frame count.

Two constraints to respect when building it.

1. **Only 4k+1 frame counts are valid.** Wan's VAE compresses time 4x with a +1
   offset, which is why the recipe uses 81 (= 4*20 + 1). Use {21, 41, 61, 81}; other
   values will not tile into latent frames cleanly, and the AR window structure
   (7 windows x 3 latent frames) is derived from the same arithmetic.
2. **`num_frames` is a dataset parameter, not an eval flag yet.**
   `multicam_pair_dataset.py` takes `num_frames=81`, but `eval_ccv_common.py` does not
   expose it. Small change: thread it through as a CLI argument, the same shape as
   `eval.py --num_input_views` on the NVS side.

Caption caveat that applies to every panel: the models were TRAINED at one setting
(8 input views for NVS, 81 frames for CCV), so points away from it measure
extrapolation, not in-distribution quality.
