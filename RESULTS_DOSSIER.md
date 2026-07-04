# Empirical Dossier: Camera Conditioning for LaCT TTT (18 evaluated runs)

Protocol (identical for all): LaCT-LVSM, 6 blocks [per-image attn(hd64) / TTT(hd256, SwiGLU x2, Muon 5, weight-norm, per-token lr) / MLP], dim 256, patch 16. RE10K 256x256, 8 input + 8 target views (train), 30k iters bs16 lr1e-4, LPIPS loss from 5k. Eval: 256 held-out scenes, 8 uniform inputs / 4 midpoint targets, PSNR/LPIPS, paired per-scene stats. Baseline PSNR 21.970, LPIPS 0.2883. Camera enters baseline ONLY via input Plucker raymap concat.

## Full results (Δ = PSNR - baseline, paired win% over 256 scenes)

| run | mechanism | Δ PSNR | LPIPS | win% |
|-----|-----------|--------|-------|------|
| **pra_h** | input line rotary (6coords x F16, 192/256 dims on q,k post-L2norm) + hidden rotary (6 x F21, 252/512 dims on h before w1, both write & read) | **+0.773** | **0.2691** | — |
| h_pra | hidden rotary only | +0.458 | 0.2716 | — |
| pra_vo | input rotary + c2w 3x3-block value/output transport | +0.427 | 0.2772 | 91 |
| pra_hi_vo | same, F=21 | +0.422 | 0.2766 | 88 |
| pra_hi | input rotary F=21 (252 dims) | +0.419 | 0.2753 | 88 |
| qk_rope_cam | input rotary F=16 (192 dims) | +0.405 | 0.2770 | 90 |
| plucker_sinc | segment-integrated 3D rotary (sinc envelope, t∈[0.05,4], 126 dims) | +0.297 | 0.2744 | 94 |
| pra_sinc | line(66pr)+segment(30pr) split budget | +0.286 | 0.2813 | 83 |
| pra_sinc_hi | same, larger budget | +0.281 | 0.2794 | 79 |
| point_rope | per-layer depth head -> 3D point rotary w/ uncertainty | +0.097 | 0.2906 | 62 |
| baseline | — | 0 | 0.2883 | — |
| q_reinject | zero-init pose bias on q only | −0.111 | 0.2914 | 25 |
| prope_ttt | projective PRoPE port (P^T/P^-1 on 128 dims + re-norm) | −0.118 | 0.2919 | 21 |
| vo_rel | value/output transport alone | −0.123 | 0.2879 | 19 |
| tour_chunk_pra | per-view chunks (8 Muon steps), far→near-target order, + input rotary | −0.227 | 0.2931 | — |
| chunk_pra | same, random order | −0.231 | 0.2951 | — |
| cam_registers | per-view camera KV registers in update set | −0.250 | 0.2928 | 9 |
| hyper_init | DeepSets camera-set -> low-rank fast-weight init delta | −0.414 | 0.2989 | 5 |
| cam_lr | camera-conditioned per-token write lr | −0.494 | 0.3007 | 5 |
| adaln_cam | per-layer zero-init pose FiLM on x | −0.605 | 0.3069 | 3 |

## R6 results (evaluated)
| run | mechanism | PSNR | Δ | LPIPS |
|-----|-----------|------|---|-------|
| pra_h_hi | input F21 + hidden F_h42 | 22.836 | **+0.866** | 0.2690 |
| pra_h_vo | pra_h + c2w value transport | 22.783 | +0.813 | 0.2681 |
| pra_sinc_h | input line+segment mix + hidden F21 | 22.711 | +0.741 | 0.2681 |
| sinc_h | input segment-sinc + hidden F21 | 22.658 | +0.688 | 0.2671 (best LPIPS) |

- F10: The HIDDEN channel is NOT saturated: F_h 21->42 = +0.09 (pra_h -> pra_h_hi). Mid-train PSNR
  suggested regression — that was batch noise; final eval wins. Agent-F's T0-scrambling *premise*
  (mid-train regression) was wrong, but h_dpra (delta-path rotation) remains live as a cleanliness fix.
- F11: sinc geometry consistently best LPIPS (0.2671) at a PSNR cost; line best PSNR. Hidden-budget
  geometry mixing (line+strat) remains a both-worlds candidate.
- F4 reconfirmed: +vo adds +0.04 on pra_h (within noise).

## Established findings

- F1 Only relative rotary addressing on q/k (and hidden h) helps. Everything else neutral-to-harmful.
- F2 Input-rotary axis SATURATES at +0.42 (F16≈F21≈+vo≈+both). More dims/freqs on the input channel add nothing.
- F3 Orthogonality is load-bearing: projective (norm-distorting) port loses to baseline. Softmax's per-row renorm protection does not exist in fast weights; norm distortion = uncontrolled write-strength perturbation.
- F4 Value payload frame irrelevant: vo alone −0.12; on top of anything +≤0.02.
- F5 Pose access at depth (FiLM/q-bias) rejected. Geometry must enter the ADDRESSING KERNEL.
- F6 Conditioning the inner optimization (lr / registers / init) all hurt. Interference with training dynamics.
- F7 Learned depth doesn't bootstrap (point_rope +0.10, depth heads stay diffuse). Budget split (line+segment) dilutes rather than helps. But pure sinc has best win% (94) and top-2 LPIPS: 3D-crossing kernel helps perceptual quality broadly with small margins.
- F8 Per-view multi-chunk updates hurt (−0.23) regardless of ordering: per-chunk weight-norm decays earlier views (recency bias), 256-token chunks below Muon amortization (~427 tok). Single big chunk is right. NOTE: full-chunk multi-STEP (update twice on the SAME 2048-token chunk) was NOT tested.
- F9 CHANNELS ARE ADDITIVE: input rotary (+0.41) ⊕ hidden rotary (+0.46) → +0.77. These relativize DIFFERENT inner-product channels of the readout expansion:
    o_j = h⁰(q_j)W1⁰ [INIT READOUT — still absolute!] + Σ lr1 ⟨h(q̃),h(k̃)⟩ v [hidden channel — relativized by h_pra] + Σ ⟨q̃,k̃⟩ c_ij [gate/hidden corrections — relativized by input rotary] + O(ΔW²)
  The ONLY remaining non-relative q-path is through the initial weights W⁰ (q̃W0⁰ inside h⁰, and h⁰(q̃)W1⁰ readout).

## Code pointers
- lact_nvs/lact_ttt.py — baseline kernel (fast_weight_swish_glu_weight_norm_mini_batch_apply)
- lact_nvs/lact_ttt_cam.py — all variants; modes combinable via "a+b"; hidden-rotary kernel copy exists (fast_weight_swish_glu_hidden_rotary_apply)
- lact_nvs/model.py — compute_camera_info (per-token Plucker in canonical frame, per-view mats), ttt_chunk_per_view/ttt_view_tour flags
- Each run: 30k iters ≈ 1.6h on one B200; 4 concurrent. Implementation must be a drop-in TTT-layer variant (config-selectable), NO backbone changes.

## GOAL UPDATE (2026-07-04)
Target raised by user: beat LaCT baseline by **+1.5 dB** (PSNR >= 23.47). Current best: pra_h_hi 22.836 (+0.87).
Comparison stays clean: only camera-conditioning changes to the TTT layer; backbone/params neutral.

## Wave 1 results (evaluated) + scope decision
| run | PSNR | Δ base | Δ vs pra_h_hi (paired t) | LPIPS |
|-----|------|--------|--------------------------|-------|
| pra_h_ms2 (2-step write +40% cost) | 22.867 | +0.90 | +0.03 (t=+3.4) | 0.2627 |
| cone_pra_h (anti-alias, F16/Fh21) | 22.741 | +0.77 | −0.10 | 0.2671 |
| budget_shift (F8/Fh31) | 22.684 | +0.71 | −0.15 | 0.2705 |
| h_dpra42 (delta-path rotation) | 22.246 | +0.28 | **−0.59 (t=−27.9)** | 0.2811 |

- F12 **Leak-fix axis is DEAD**: routing the init readout around the hidden rotation loses 0.59 dB.
  The rotated init readout (R_j h)W1^0 is not pollution — it is a *functional view-dependent prior*
  that the slow weights exploit. w0_mask/cone_dpra killed mid-run; steer_glu deprioritized.
- F13 ms2: statistically significant but tiny (+0.03) at +40% TTT cost → perfect control experiment:
  "2x deeper writes buy what the free embedding already bought." Out of main line per user direction
  (goal = optimal positional embedding at minimal overhead).
- Aliasing (cone) neutral at F16; input budget below F16 loses (budget_shift).
- Scope reset by user: positional-embedding-only, minimal overhead. Dynamics cards (ms2/res2/loo) out.

Now training (embedding-only): cone_hh (cone F21·64pi + h F42), mip_hh (per-layer half-octave stagger,
F21/F42), omega_map (learnable 6->P phase maps, F16/F21 base), m_scale (per-scene moment whitening).

## Wave 2 results (embedding knobs)
| run | PSNR | Δ base | vs own base | LPIPS |
|-----|------|--------|-------------|-------|
| mip_hh (layer stagger, F21/42) | 22.862 | +0.89 (record) | +0.03 vs pra_h_hi | 0.2672 |
| cone_hh (F21/42) | 22.845 | +0.87 | +0.01 neutral | 0.2686 |
| omega_map (F16/21 base) | 22.793 | +0.82 | +0.05 vs pra_h | 0.2675 |
| m_scale (F16/21 base) | 22.664 | +0.69 | −0.08 REJECTED | 0.2727 |
- F14: embedding knobs now yield +0.03..0.05 each — second saturation plateau near +0.9.
Wave 3 (running): stack1 = mip+omega_map (F21/42); stack2 = cone+mip+omega_map; omega_hh (attribution
control: omega_map alone at F21/42); h_strat (depth-stratified hidden kernel — last untested geometry).

## Wave 3 results
| run | PSNR | Δ base | LPIPS |
|-----|------|--------|-------|
| omega_hh (omega_map alone, F21/42) | 22.901 | **+0.93** (record) | **0.2651** (record) |
| stack1 (mip+omega) | 22.898 | +0.93 | 0.2663 |
| stack2 (cone+mip+omega) | 22.805 | +0.84 | 0.2704 |
| h_strat (depth-sliced hidden) | 22.602 | +0.63 | 0.2743 |
- F15: learnable phase maps (omega_map) = only knob that added on champion (+0.065). Stacks NOT
  additive (mip+omega = omega); cone slightly negative in stack; h_strat rejected (line kernel wins).
- F16: embedding-knob axis plateaus at ~+0.93. User scope decision: TTT-layer only (no attention ext).
Wave 4 (running): omega_hh seeds 137/211, baseline seed 137 (seed variance for the delta claim),
omega_r (random-tilt dOmega init).

## Wave 4 results — omega_r record + seed-variance discovery
| run | PSNR | LPIPS |
|-----|------|-------|
| omega_r (tilt 0.1 init, F21/42) | **23.010** | **0.2592** (both records) |
| omega_hh_s3 (seed 211) | 22.932 | 0.2623 |
| omega_hh_s2 (seed 137) | 22.813 | 0.2649 |
| baseline_s2 (seed 137) | **21.617** | 0.2963 |
- F17: random-tilt dOmega init beats zero-init by +0.11 (zero-init stays near axis alignment).
- F18: **baseline seed variance is large** (21.970 vs 21.617, spread 0.35). Fair mean-vs-mean deltas:
  omega_hh(3 seeds, 22.882±0.05) - baseline(2 seeds, 21.794) = **+1.09**. Single-seed comparisons vs
  the lucky baseline seed have been UNDERSTATING the method. Champion variance much smaller than baseline's.
Wave 5 (running): omega_r seeds 137/211, baseline_s3 (seed 211), omega_r2 (tilt 0.2).

## Wave 5 results — fair 3-seed statistics
| | seeds | mean PSNR | std | mean LPIPS |
|---|-------|-----------|-----|------------|
| omega_r (tilt 0.1) | 95/137/211 | **22.971** | 0.088 | 0.2613 |
| baseline | 95/137/211 | 21.745 | 0.196 | 0.2929 |
| omega_r2 (tilt 0.2, 1 seed) | 95 | 23.049 | — | 0.2622 |
- **Fair mean-vs-mean delta: +1.226 dB** (and −0.032 LPIPS). Method is 2x more seed-stable than baseline.
- Tilt 0.2 > 0.1 on seed 95 (23.049 vs 23.010).
Wave 6 (running): omega_r2 seeds 137/211; omega_r3 (tilt 0.3); omega_rb (tilt 0.2 + learnable
per-pair phase bias — cancels in differences, re-frames the functional W^0 absolute-phase path per F12).
