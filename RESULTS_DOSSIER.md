# Empirical Dossier: Camera Conditioning for LaCT TTT

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
- Each run: 30k iters ≈ 1.6h on one B200; 8 concurrent (8-GPU batch node since 2026-07-09). Implementation must be a drop-in TTT-layer variant (config-selectable), NO backbone changes.

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

## Cross-task validation: LLM (lact_llm, 200M params, 3B tokens fineweb-edu, matched data order)
| variant | val loss | ppl | Δ vs original |
|---------|----------|-----|---------------|
| base_nope (no fw pos enc) | 2.9735 | 19.56 | +0.0125 |
| base_rope (original: fw-RoPE on) | 2.9610 | 19.32 | 0 |
| **h_pra (hidden rotary, 1D)** | **2.9513** | **19.13** | **−0.0097 (−1.0% ppl)** |
| full (+learnable freqs) | 2.9628 | 19.35 | +0.0018 |
- F19: h-PRA generalizes to language modeling: gain equals the entire nope→rope gap, additive on top
  of it, stable from step 4k to 91k. Original authors' "fw-RoPE ~ NoPE" observation reproduced.
- F20: omega_map degenerates in 1D (no direction to learn) — its power is multi-dimensional phase
  direction learning (6D NVS: +0.11; 1D LLM: ~0). Boundary condition for the paper.

## Cross-task validation: AR video (Wan1.3B attn-only finetune, MultiCamVideo, 4100 steps, deterministic noise)
Paired per-step loss over last 2500 steps: h_pra Δ=+0.000000 (t=0.0), full Δ=+0.000027 (t=1.2, n.s.)
- F21: h-PRA NEUTRAL on short-budget video finetune (no gain, no harm). Caveats: 4100 samples at
  batch 1; 6 write chunks/seq (7 AR windows x 3 latent frames, interleaved noisy/clean; write on
  clean windows only, last unused). CORRECTED 2026-07-05: earlier "~2 chunks" was an estimate;
  exact count from the ar_lact_swa_repeat chunk loop is 6. SWA window = 1 full AR window (4680
  tok), so memory-ONLY content is >=2 windows back -- thin for natural video (adjacent-frame
  redundancy). Real story: memory-exclusive workload share, not write count (video 6 > LLM 4 >
  NVS 1 writes, yet gains go the other way); noisy diffusion objective.
  Honest verdict: 2 of 3 tasks improve (NVS +1.22 dB, LLM −1.0% ppl), video unaffected.

## F22: v20k video ablation final (20,000 steps, completed 2026-07-07)
Paired per-step loss (deterministic noise, n=2090 common log points):
- h_pra - base: second half (10k-20k, n=1001) -0.000001 (t=-0.1); final 2.5k +0.000002 (t=+0.2).
  VERDICT: h-PRA exactly neutral at 5x the F21 budget. Boundary condition confirmed, not
  budget-limited. No gain, no harm.
- full - base: second half +0.000022 (t=+2.8); final 2.5k +0.000033 (t=+2.6). Statistically
  detectable, practically negligible (+0.04% of loss): learnable hidden freqs drift without
  useful signal in this regime. Honest note for paper if full variant is mentioned for video.
Paper Sec 4.3 (boundary condition framing) stands unchanged.

## F23: Q1 absolute-adaptation probe (q1_scenerand, seed 95, completed 2026-07-07)
Design: full PRA recipe (qk_rope+h_pra+omega_map, F21/42) but phase coords replaced by ONE
random 6-vector per scene (resampled every forward; relative rotations all identity; raymap
inputs stay true rays).
Result: PSNR 21.634 +- 0.145 (eval stderr), LPIPS 0.3060 vs baseline 21.745 +- 0.196 (3-seed)
/ 0.2929 and full PRA 22.971 +- 0.088 / 0.2613.
- PSNR: within noise of baseline (-0.11, < 1 baseline seed-std). Even adversarially random
  absolute stamps, resampled per visit, are absorbed by the slow weights. The benign-residue
  claim ("What stays absolute") now has direct experimental support beyond F12.
- LPIPS slightly worse than baseline (+0.013): small but visible perceptual cost of pure
  absolute perturbation. Honest caveat for the paper.
- Relative isolation: full PRA - q1_scenerand = +1.34 dB PSNR / -0.045 LPIPS: essentially the
  entire PRA gain is carried by the relative component.

## F24: Q2 depth-3 fast weights, one rotary per address space (seed 95, 2026-07-07)
| variant | PSNR | LPIPS | note |
|---|---|---|---|
| fw3l_base (depth-3, no rotary) | 21.868 +- 0.143 | 0.2932 | ~= 2L baseline 21.745: depth alone adds nothing |
| fw3l_rot2 (input + s2 sites) | 23.307 +- 0.161 | 0.2517 | already beats 2L record recipe |
| fw3l_rot3 (all 3 sites) | 23.439 +- 0.161 | 0.2478 | NEW RECORD (single seed) |
Paired per-scene: rot3 vs rot2 +0.132 dB (t=+15.2, win 86%); rot3 vs base +1.571 (t=+31.4).
- "One rotary per address space" VALIDATED: the third site earns its rotation at +1.5k params.
- Depth-3 alone is worthless (+0.12, noise) but depth-3 + full addressing = best result to date:
  the ViT3 "deep inner models don't help" observation is an ADDRESSING failure, not a depth
  failure. Strong candidate for a paper subsection.
- Stability: no lr retune needed; Muon + weight-norm on all 4 matrices held at base_lr 0.01.
- Kernel verified bit-exact vs autograd (base/rot2); 3-seed replication launched (137, 211).

## F24b: Q2 3-seed replication (seeds 95/137/211, 2026-07-07)
| variant | PSNR (3-seed) | LPIPS (3-seed) |
|---|---|---|
| fw3l_rot3 | 23.439 +- 0.022 | 0.2478 +- 0.0005 |
| fw3l_rot2 | 23.301 +- 0.015 | 0.2518 +- 0.0007 |
- NEW HEADLINE CANDIDATE: fw3l_rot3 = +1.694 dB over 2L baseline (21.745 +- 0.196),
  +0.47 over the previous record recipe (22.971 +- 0.088), at 3-seed rigor.
- Third-site gain replicates in every seed (per-seed rot3-rot2: +0.132/+0.106/+0.178).
- Rotary runs are far more seed-stable (std 0.02) than the baseline (std 0.196).

## F25: environment reproducibility + Q4 main ablation (rebuilt env, 2026-07-09)
Node reset wiped the old env (conda, /tmp data, ALL checkpoints). Rebuilt from scratch:
venv torch 2.11+cu128 (B200/sm_100), RE10K reshared from surviving lustre source.
Reproduction check (same protocol, seed 95, new env vs dossier):
| run | old | new | delta |
|---|---|---|---|
| fw3l_rot3 | 23.439 / 0.2478 | **23.439** / 0.2483 | 0.000 / +0.0005 |
| pra_h_hi | 22.836 / 0.2690 | 22.797 / 0.2685 | -0.039 |
| pra_hi | 22.389 / 0.2753 | 22.333 / 0.2751 | -0.056 |
- Env change (torch 2.4->2.11, cu124->cu128, conda->venv) is result-neutral: headline
  reproduces to the third decimal; others within seed-scale noise. All dossier numbers remain valid.

Q4 fixed-ladder ablation (input F21 / hidden F_h42, no learnable freqs; 3 seeds 95/137/211) — COMPLETE:
| variant | PSNR (3-seed) | LPIPS | delta vs base 21.745+-0.196 |
|---|---|---|---|
| full (pra_h_hi) | 22.824 +- 0.065 | 0.2664 +- 0.0024 | **+1.079** |
| w/o input (h_pra_hi) | 22.701 +- 0.154 | 0.2677 +- 0.0031 | +0.956 |
| w/o hidden (pra_hi) | 22.348 +- 0.033 | 0.2763 +- 0.0010 | +0.603 |
- Hidden channel carries most of the fixed-ladder gain (+0.96 of +1.08); input adds +0.12 on top.
- Channels sub-additive at saturated ladders (0.60+0.96=1.56 > 1.08 actual), unlike the small-ladder
  additivity of F9 (F16/F21: 0.41+0.46 ~= 0.77): at F21/F42 the two address spaces partially overlap.
- h_pra_hi (hidden-only) is a NEW variant: strongest single-site fixed-ladder recipe.
- Input-only is the most seed-stable of the three (std 0.033) but the weakest: the input channel
  saturates early (F2) AND overlaps with what the hidden channel already delivers.

## F26: gateless 2-layer-MLP fast weights (Q6, seed 95, 2026-07-09)
Inner model f(x) = silu(x W0) W1 (SwiGLU gate branch removed), inter_multi 3 for exact
fast-weight param parity (393,216); kernel autograd-verified. Sites: input F=21, hidden F_h=64
(d_h=768 budget rule).
| variant | PSNR | LPIPS |
|---|---|---|
| mlp2_base | 20.532 +- 0.135 | 0.3367 |
| mlp2_rot2 | 22.465 +- 0.149 | 0.2722 |
- Rotary gap on the plain MLP: **+1.933 dB** — larger than SwiGLU's fixed-ladder +1.08. The
  recipe transfers unchanged to the textbook fast weight; second external validation of
  "one rotary per address space" (after fw3l, F24).
- mlp2_rot2 (22.465) overtakes SwiGLU-base + input-only rotary (pra_hi 22.348) despite a
  1.2 dB weaker base: addressing quality dominates inner-model capacity.
- Secondary: the SwiGLU gate itself is worth +1.21 dB of base capacity at equal params
  (contrast F24: extra depth was free but worthless without addressing).
- Single seed; seeds 137/211 queued for free GPUs.
