# Experiment Queue

Pending experiments, in priority order. NVS small config = L6/d256/p16, 30k
iters, bs16, ~1.6h per run on one B200. GPU 3 is free while ccv occupies 0-2
(until ~2026-07-09); cheap NVS items can run there anytime.

## Q1. Absolute-adaptation probe (per-scene-constant random phases)
Requested 2026-07-07 (paper Sec "What stays absolute" support).
- Design: full TTT-RoPE recipe, but replace every token's phase coordinates
  pi with ONE random 6-vector drawn per scene (same for all tokens in the
  scene; resampled every scene, train and eval). All relative rotations
  become identity; only the absolute stamps vary across scenes. Raymap
  INPUT features stay the true rays (only the rotary phases are randomized).
- Compare: (a) baseline no-rotary, (b) this variant, (c) full PRA.
- Reading: (b) vs (a) measures how well slow weights absorb pure absolute
  phase variation (benign-residue claim: expect (b) ~= (a), no collapse);
  (c) - (b) isolates the purely relative part of the gain. Complements F12
  (h_dpra: cancelling the rotated init readout HURT -0.59).
- Cost: 2 new runs (b is new; a, c exist) = ~3.2h on GPU 3. Implementation:
  one flag in lact_nvs (cam_phase_override=scene_random), seed-stable per
  scene id.

## Q2. Deeper fast weights: one rotary per address space at depth 3
Requested 2026-07-07 (Method "Generalization to other fast weights" claim).
- Design: replace the SwiGLU fast weight with a 3-layer inner network
  (e.g., W3' silu-gate stack or plain W_out sigma(W_mid sigma(W_in x))) in
  the NVS small config; sites = input, hidden1, hidden2. Runs:
  (a) 3L baseline no rotary, (b) 3L + 2 rotaries (input + last hidden),
  (c) 3L + 3 rotaries (all address spaces).
- Reading: does the third interface earn its rotation ("one rotary per
  address space")? Watch inner-loop stability (LaCT weight-norm on the
  middle matrix; may need lr retune) - ViT3's insight 5 says deep inner
  models are fragile, our stabilizers may be the difference.
- Cost: 3-4 runs x ~2h (deeper inner model is slower). After Q1.
- Note: also the theory check for the ViTTT Stage-1 plan (hidden-site
  creation), see chat 2026-07-06.

## Q3. (carried) ViT3 Stage-1: TTT-RoPE on plain ViT3-T
- 4 runs, ImageNet-1k 100 epochs: baseline / +input rotary (2D coords) /
  +W_out 2-layer inner / +W_out + hidden rotary. Needs ImageNet download.
- After ccv completes (GPU budget).

## Done / superseded
- v20k long-budget video ablation -> F22 (h-PRA exactly neutral at 20k).
- ccv 3-run grid (base/pra/both) RUNNING on GPUs 0-2 since 2026-07-07.
