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
