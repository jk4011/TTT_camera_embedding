# Q28 ZipMap evaluation protocol (defined 2026-08-04, node1)

Host: **ZipMap** — "Linear-Time Stateful 3D Reconstruction via Test-Time Training", CVPR 2026
(Google DeepMind + Cornell + MIT), arXiv 2603.04385. Memory is a near-verbatim LaCT
`FastWeightGluMLPMultihead` (same class name as ours), 24 global TTT blocks,
w0[1,1024,2048] / w1[1,2048,1024] / w2[1,1024,2048], 1.40B params total.

**Why this host**: its *state query* reads out RGB **and depth** at a target camera pose from ray
maps ALONE — `render()` is documented "render nvs rgb and depth given ray conditions only". A query
with no image content in it cannot be resolved by content matching, so it is the purest
coordinate-addressed retrieval available in any public model. This is the property that made every
other candidate weaker (FSM/tttLRM = posed NVS, i.e. our existing task; CUT3R/TTT3R = no hidden
site).

---

## 1. Cells (seed-matched, identical data order, one flag apart)

| cell | rotary | what it tests |
|---|---|---|
| `base` | none — stock ZipMap (Plucker ray maps concatenated to input tokens) | the incumbent, and our control |
| `in`   | `qk_rope_cam` on fast-weight q/k | input-site addressing |
| `h`    | `h_pra` on the SwiGLU hidden | **the site nothing in the literature occupies** |
| `both` | both | additivity |
| `content` | re-enable ZipMap's dormant `posed_ray` path: pose injected as CONTENT, no rotation, `input_ray_patch_embed.proj` zero-init so it starts exactly at the released checkpoint | **REQUIRED, not optional** (see sec. 6) — the information-matched control that separates "GT pose helps" from "addressing beats content" |

All cells: same seed, same data order, same budget, same everything else. Fixed frequency
ladders only (learnable is negative on every task we have run: F33/Q20/Q21/F37).

## 2. Data

**Train (fine-tune)**: DL3DV train at `V-LAB/Datasets/dl3dv/dl3dv_undistorted_960/train`,
**filtered to exclude the 130 scenes that also appear in test** -> 9,995 scenes. Verified locally:
`comm -12 <(ls test) <(ls train)` = 130. Scene format `images_undistort/` + `opencv_cameras.json`
is ZipMap's native layout, so no conversion.
If GT depth is needed for the depth metric, add ONE synthetic set with depth (VKITTI or MVS-Synth;
both have download+preprocess scripts in the repo) — download to `datasets/zipmap/`.

**Eval, primary**: DL3DV test (140 scenes), clean once train is filtered.
**Eval, cross-set**: RE10K test (already resharded locally). RE10K is **not among ZipMap's 29
training datasets**, so it also checks that any gain is not DL3DV-specific.

## 3. Primary metric — state-query readout at held-out poses

For each eval scene: feed N input views to build the fast-weight state, then query at K held-out
target poses using **ray maps only**. Report:
- **RGB readout**: PSNR / SSIM / LPIPS (DL3DV + RE10K; no GT depth needed).
- **Depth readout**: AbsRel and delta<1.25 (only on the synthetic set that has GT depth).
Fix N, K, and the view-sampling rule ONCE and pin them in a split file — ZipMap ships no fixed
eval split (`index_plan` is unused), and FSM's per-scene PSNR ranged 22.5-36.7, so unpinned
sampling is worth several tenths of a dB. Pin seed 42.

## 4. Statistics (house standard)

**Per-scene paired deltas with t-statistics**, cells seed-matched — never compare unpaired means.
Single-seed gaps at NVS scale of 0.1-0.3 dB are init noise (F18), so any headline claim needs
either a large paired t or a second seed. Report win-rate (scenes better / total) alongside the mean.

## 5. Controls that must pass before any number is believed

1. **Zero-phase equivalence**: with rotary gains zeroed, output must equal stock ZipMap to
   ~0.000e+00 (the FSM port achieved exactly this; the harness transfers).
2. **Anchor reproduction**: reproduce one *published* ZipMap number with the released checkpoint —
   the RE10K relpose AUC row of Table 1 is the target, since RE10K is untouched by their training
   and we hold it. This validates our harness before we trust our own cells.
3. **Pose-shuffle control**: randomize the query poses at eval. If our gain is genuine coordinate
   addressing, shuffling must destroy the rotary cells' advantage MORE than the base cell's. A gain
   that survives pose shuffling is not coming from the mechanism we claim.
4. **All apply sites patched**: the SwiGLU apply is duplicated at three places in ZipMap's
   `ttt.py` (132, 134-137 with the inverse rotation, 167). FSM had the same trap. Verify by asserting
   the fused path off and diffing outputs across code paths.

## 6. Open design decision (gates the whole thing)

Our phases need camera poses at fast-weight UPDATE time, but ZipMap's default mode is **unposed**
(poses predicted by `camera_head`). Options, in preference order:
(a) **pose-conditional mode** if one exists (the `posed_ray` path exists in the model but the
    trainer raises for it) -> phases from ground-truth relative pose, exactly like NVS/CCV. Clean.
**RESOLVED 2026-08-04**: there is NO pose-conditional input mode in anything ZipMap released.
Verified four ways: `input_ray_patch_embed` has 0 keys in all four checkpoints; no config sets
`nvs_input_type`; `trainer.py:1440` raises "Only 'unposed_ray' is currently supported"; the paper
computes the ray map from the TARGET camera only. `posed_ray` (aggregator_ttt.py:137-145, 259-264)
is untrained dead code with a latent key-name bug (ZipMap.py:196 reads `input_ray_cond`, the
trainer writes `nvs_input_ray_cond`).

**This does not block us**, because our method injects pose as ADDRESSING (phases computed outside
the network from poses and threaded through `info` into the rotary), not as content. The rotary
cells need NO new modules and NO revival of `posed_ray`: zero new parameters, and the converted
model still strict-loads the released checkpoint (only `freq_gain`/`gain_h` are extra).

DECISION: **(a) give GT poses for the input views** in the main grid. Faithful test of the
inner-product addressing lemma, matches our NVS/CCV setting. Freeze `camera_head` (it becomes
near-redundant) and disclose this.
CONSEQUENCE: because only the rotary cells would otherwise receive GT pose information, the
**pose-as-CONTENT cell becomes mandatory** — without it a win confounds "GT pose helps" with
"addressing beats content". Hence 5 cells, with content-vs-rotary as the load-bearing comparison.

REJECTED — **(b) query-side-only rotation is UNSOUND** (this corrects an earlier draft of this
document). If `k` (update tokens) is unrotated and only `q` is rotated, `q.k` carries the query's
ABSOLUTE phase — the phases do not cancel, so it is pose-modulated gating, not relative positional
encoding, and it cannot test the lemma. Keep only as a deliberately-broken ablation predicting null.
REJECTED — (c) GT at train / predicted at test: train/test mismatch.
DEFERRED — (a2) two-pass with predicted input poses (run trunk, read `camera_head`, re-rotate with
detached predictions): preserves the unposed selling point at ~2x forward cost. Use as a robustness
FOLLOW-UP on the winning cell ("does the effect survive predicted poses?"), not as the primary grid.

## 6b. Measured scale (one B200, released state-query config, 1.401B params, 518^2, bf16 + act-ckpt)

| views/GPU (input+query) | s/step | peak mem |
|---|---|---|
| 12 (8+4) | 0.71 | 30.6 GiB |
| 24 (16+8) | 1.24 | 43.7 GiB |
| 44 (32+12) = recipe max | 2.17 | 65.8 GiB |

Per-GPU load sets the step time, so **1 GPU per cell with cells CONCURRENT is 4x better wall-clock
than one cell on 4 GPUs**. At 24 views/43.7 GiB two cells fit per GPU, so **all 5 cells fit on
4 GPUs in one wave**. Plan: 5 cells x 24 views x 10k iters, fine-tuned from
`checkpoint_with_ref_view.pt` (stage 2 — exactly what upstream fine-tunes the query model from)
=> ~4-8 h per seed round, ~9-17 h for two matched seeds. (Full published recipe is 100k steps x
8 GPUs ~= 2.9 days PER CELL — infeasible.)
Only `checkpoint_state_query.pt` has the query pathway (ray_patch_embed + 62 nvs_head keys); the
other three have zero NVS keys. `checkpoint_online.pt` uses `camera_mlp_head`, not swap-compatible.

## 7. What survives at reduced scale

Full ZipMap training is out of reach. We fine-tune from a released checkpoint at reduced iters on
DL3DV. That supports: "**rotary addressing can be grafted onto a concat-trained spatial memory and
improves its pose-queried readout**" — a paired, matched-budget claim. It does NOT support
"rotary is better when trained from scratch at scale", and absolute numbers are not comparable to
their published tables (different data, different budget). Say so explicitly in any writeup.
