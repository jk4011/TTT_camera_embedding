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
| variant | PSNR (3-seed 95/137/211) | LPIPS (3-seed) |
|---|---|---|
| mlp2_base | 20.500 +- 0.041 | 0.3357 +- 0.0022 |
| mlp2_rot2 | 22.477 +- 0.099 | 0.2723 +- 0.0032 |
- FINAL (3-seed complete 2026-07-10; per-seed gaps +1.93/+2.07/+1.93): rotary gap on the
  plain MLP = **+1.977 dB** — larger than SwiGLU's fixed-ladder +1.08. The recipe transfers
  unchanged to the textbook fast weight; second external validation of "one rotary per
  address space" (after fw3l, F24).
- mlp2_rot2 (22.477 +- 0.099) overtakes SwiGLU-base + input-only rotary (pra_hi
  22.348 +- 0.033) despite a 1.2 dB weaker base: addressing quality dominates inner-model
  capacity. Both at 3-seed rigor now.
- Secondary: the SwiGLU gate itself is worth +1.25 dB of base capacity at equal params
  (21.745 vs 20.500; contrast F24: extra depth was free but worthless without addressing).

## F27: LLM 2x2 input/hidden rotary grid, rebuilt env (Q7, 2026-07-09)
200M LaCT LM, 3B tokens fineweb-edu (fixed data order, data_seed 42), bs 8x4096, 91,552 steps;
all four runs share the SAME kernel path (ttt_prenorm=True, use_fused_kernel=False), the same
val cache (2M tokens, built once before all runs), flags verified from each run's logged config.
| val ppl | hidden OFF | hidden ON |
|---|---|---|
| input fw-RoPE ON | **18.40** (abl_rope) | 18.64 (abl_hpra) |
| input fw-RoPE OFF | 18.62 (abl_nope) | 18.85 (abl_honly) |
- Clean additive 2x2: input fw-RoPE = −0.22..−0.21 ppl in both rows (replicates old-env F19
  −0.24); hidden rotary = +0.23..+0.24 ppl in both columns — in the rebuilt env the 1D hidden
  rotary consistently HURTS, incl. the new hidden-only cell.
- SIGN FLIP vs old-env F19 (h_pra 19.13 vs rope 19.32 = −0.19 help). Same code (no commits to
  lact_model since 7559589, which produced F19); differences are env-level only (new fla/torch,
  re-downloaded streaming dataset, rebuilt val cache — absolute ppl level also shifted ~−0.9).
- CODE AUDIT (user-requested, 2026-07-09): hidden-rope kernel re-verified numerically —
  (a) zero-angle: bit-exact equality with the baseline kernel (max diff 0.0e0);
  (b) rotated manual gradients (dw0/dw1/dw2 incl. inverse-rotation backward) match torch
  autograd to 7e-9 (fp32). Kernel diff vs baseline is rotation-only (loop bounds, muon,
  weight-norm, momentum, tail chunk identical). No bug found; the flip is empirical.
- Honest reading: the 1D hidden-rotary effect is small and NOT robust across environments
  (−0.19 old vs +0.24 new, both single-seed/single-env-instance); the robust LLM finding is
  input fw-RoPE (−1.2% ppl in both envs). Contrast NVS F25 where hidden carries +0.96 of
  +1.08 at 3-seed rigor: relative addressing in the hidden space pays where the coordinate
  is multi-dimensional (6D rays), not in 1D text. Paper framing: report input-RoPE as the
  LLM result; treat hidden-1D as boundary/unstable (like video F21/F22) pending seed
  replication (rope+hpra seed-2 would settle it; ~2x12h).

### F27b: deep audit of the sign flip (user-requested, 2026-07-09) — no bug; mechanism identified
Everything below rules candidate causes in or out:
1. INDUCTOR RULED OUT: training executes @torch.compile'd kernels; compiled-vs-eager at real
   shapes/dtypes (B=32, L=4096, d=192, chunk 1024, muon+momentum, bf16) gives mean-rel diff
   2.43e-2 (baseline) vs 2.47e-2 (hidden-rope), ratio 1.02, both bit-deterministic run-to-run:
   inductor treats both kernels identically; no hidden-rope-specific miscompilation.
2. HYPERS RULED OUT: train_small.py diff since the F19 launch commit = cache dirs + os._exit
   only; chunk 1024, window 1024, rope_theta 1e6, tilt, muon, momentum all identical.
3. DATA SNAPSHOT RULED OUT: fineweb-edu lastModified 2025-07-11, tokenizer repo 2025-02 —
   both predate ALL runs; same corpus, same tokenizer content.
4. REMAINING MOVING PART = `datasets` streaming shuffle implementation (old env version
   unknown/lost with T_B): same seed, different library version => different stream order =>
   different 3B training subset AND different 2M val sample. This is the only surviving
   explanation for the −0.9 absolute ppl level shift, and it makes old-vs-new deltas
   different-data-draw comparisons (within-env comparisons stay clean).
5. MECHANISM PROBE (checkpoint surgery + per-position val loss, 512-token buckets):
   - The hidden-ON deficit is FLAT in position (+0.011..+0.015 loss in every bucket, both
     rows) — present already in bucket 0 where no fast-weight updates exist yet. So it is
     NOT long-range recall attenuation; it is a constant tax, i.e., the absolute-phase
     burden of serving position-rotated hidden activations through the initial readout.
   - Surgery asymmetry: hpra with rotation zeroed at eval loses only +0.003 loss (the
     trained model largely IGNORES the hidden rotation when input RoPE exists); honly with
     rotation zeroed collapses at early positions (18.85 -> 22.51 ppl, damage confined to
     the first ~1.5k tokens) — without input RoPE the model does use the hidden rotation
     as its only positional signal, yet still nets worse than nope.
   - Cross-injection control: rotation forced ON at eval for rope/nope models = 27.8/94.3
     ppl with position-growing damage (expected; addresses scrambled).
   CONCLUSION: in 1D the hidden rotary buys ~nothing relative (content addressing suffices;
   position is already covered by input RoPE + attention) but pays the constant absolute-
   phase tax => small net negative, with sign sensitive to the data draw (explains old-env
   +: different 3B subset/val sample sat on the other side of a ~1% effect). NVS is the
   opposite regime: the coordinate is 6D and relative geometry IS the signal, so the same
   rotation earns +0.96 dB at 3-seed rigor.

### F27d: ds43 pair — the hidden-1D sign FLIPS BACK on a different draw/budget (2026-07-10)
rope vs hpra at data_seed 43, 0.5B tokens (15,258 steps, full cosine), same protocol
otherwise (val = ds43 stream head, shared by the pair):
- rope_ds43 3.2906 / ppl 26.86; hpra_ds43 3.2808 / ppl **26.60** -> hidden rotary HELPS
  by −0.26 ppl (loss −0.0098 — almost exactly the old-env F19 gain, −0.0097).
- So across three measurements of the same design: old env 3B: −0.19 (help);
  new env ds42 3B: +0.23 (hurt); new env ds43 0.5B: −0.26 (help). The effect is real
  but its SIGN depends on the data draw and/or token budget — consistent with F27b's
  "~1% effect sitting on a boundary". Note the ds42 full run's own curve: honly−nope was
  NEGATIVE (helping) at steps 3k–10k and only turned positive after ~11k, i.e., the
  hidden rotary tends to help EARLY and cost LATE in ds42 as well.
- CONFOUND: ds43 pair ran at 0.5B (launched before the GA protocol was fixed), so draw
  and budget are entangled. Disentangler launched (gpu1): ds42 rope+hpra at the SAME
  0.5B budget -> completes the {draw} x {budget} 2x2. GA gen-0 (0.65B, ds42) adds the
  honly-row short-budget point.
- Implication for Q9 (user's push to make honly win): at short budgets the hidden
  rotary already CAN win in 1D; the enemy is late-training erosion (consistent with the
  F27b tax story: the absolute-phase tax is constant while the relative/prior benefit
  saturates or is learned around).

## F28: Q9 GA verdict — a low-gain hidden ladder makes hidden-only WIN in 1D at 3B (2026-07-10)
User-driven program ("honly도 이길 수 있을 것"): evolve honly variants. Genes implemented:
ttt_hrope_frac / gain / theta / delta_only (audited kernels; ledger lact_llm/ga_honly/).
HEADLINE (3B, full F27 protocol, ds42): **honly + ladder gain 0.1 = 18.53** vs nope 18.62
(−0.09; gap stable −0.09..−0.10 across 53k-91.5k checkpoints) vs plain honly 18.85 (+0.23).
The gentle ladder converts the hidden-only deficit into a gain: swing +0.32 ppl. It also
beats hpra 18.64. Input rope alone (18.40) remains best.
STACKING RESULT (recorded 2026-07-13; the run finished in the overnight batch but was
left out of the write-up): **ga_hpra_gain01_full (input rope + gain-0.1 hidden) = 18.415**
vs rope 18.405 — the gentle ladder removes the old stacking TAX entirely (plain hpra
18.639 -> 18.415, -0.22) but adds NO increment over input alone (+0.01, noise). Best
hidden recipe, same verdict: in 1D the hidden site earns nothing on top of the input
site — mirrors the frozen-video F31 both-vs-in (+0.06%, n.s.); the hidden increment
only appears in full-training multi-D video (F30, t=-9.0). Completed LLM grid (3B ds42):
nope 18.620 / rope 18.405 / honly-g0.1 18.53 / hpra-g0.1 18.415 / hpra-plain 18.639.
CONFIRMATIONS (overnight 3B batch, 2026-07-10/11):
- Seed replicate COMPLETE (rerun after the 75k hf-streaming crash): s137 endpoint
  gain01 18.64 vs nope 18.68 = −0.04. Two-seed summary: −0.09 (s42) / −0.04 (s137),
  mean −0.07 — sign replicates on both seeds; honest wording: a small but consistent
  gain (3B endpoint seed noise is ~0.06, so individual endpoints are marginal; the
  matched-step trajectories are the stronger evidence).
- gain 0.03 at 3B: 18.70 (+0.08 vs nope) — 0.03 LOSES at decision budget; 0.1 is the
  scale (proxy preference for 0.03 was noise, as predicted by the 4-cell analysis).
- STACKING: rope + gain-0.1 hidden = 18.41 vs rope 18.40 (+0.01, neutral). The gentle
  hidden is redundant given input rope — but note plain hidden on rope COST +0.24
  (hpra 18.64); the gentle ladder also removes the harm. Practical ranking in 1D:
  input rope (−0.22) > honly gentle hidden (−0.09) > nothing; hidden adds nothing on top
  of rope.
Supporting proxy-scale findings (20k, 0.65B):
- Gain line search (ds42 s42): 1.0: 26.10 / 0.1: 25.68 / 0.03: 25.50 / 0.01: 25.92 /
  ->0 (=nope): 26.02 — interior optimum: a SLOW coarse position signal (top ladder period
  ~200 tokens at gain 0.03) is genuinely better than none.
- frac 1.0 at gain 0.03: 26.10 — REJECTED. Rotating all hidden dims kills the gain even at
  slow frequencies: the position-free half of the hidden space (pure content pathway) is
  load-bearing. Matches the F27b picture: keep a tagged subspace AND an untagged one.
- delta_only: beat plain ctrl (25.80 vs 26.10 s42) but subadditive with low gain.
- METHODOLOGY: 20k-proxy single-run gaps of +-0.3 ppl are init-seed noise (measured 4-cell
  gain01-nope gaps: −0.34/+0.06/−0.16/+0.21, mean −0.06 +- 0.12 SE). Fine-grained gain
  distinctions (0.03 vs 0.1) are unresolved at proxy scale; only the 3B trajectory-stable
  comparison is decision-grade. (LLM analogue of F18.)

## F42: the rotary's wall-clock cost was 50x its bandwidth floor; fusing it cuts the
## overhead from 42% to 9% (2026-08-05)
The rotary's arithmetic is trivial, yet `both` ran 42% slower than `base` on tttLRM.
Profiled at the real hidden shape [4, 4096, 3072] on a B200:
| implementation | time | vs bandwidth floor |
|---|---|---|
| memory-bandwidth floor (8 TB/s, 288 MiB traffic) | 0.038 ms | 1x |
| original | 1.892 ms | **50.1x** |
| removing the redundant stack/cat only | 1.794 ms | 47.5x (worth 5%) |
| **fused into one kernel** | **0.160 ms** | **4.2x (11.8x faster)** |
CAUSE: not FLOPs. Each elementwise step was its own kernel launch, so the 288 MiB of
traffic was paid six or seven times. Fusion pays it once.
WHY IT HAD NEVER BEEN FUSED: the launchers exported `TORCH_COMPILE_DISABLE=1`, which
disables compilation *globally*. It was set for ONE function,
`fast_weight_swish_glu_weight_norm_mini_batch_apply`, whose `sp_all_reduce` carries a
ProcessGroup that inductor cannot trace. Removing `@torch.compile` from just that
function (and its cam twin) lets everything else compile.
Also removed per-block recomputation of the phases: `gain` is a plain float and `omega`
a fixed buffer, so all 24 blocks were rebuilding identical cos/sin. At nf_h=256 and 16
views that is ~36 GB of redundant allocation per forward. Now memoised in `info`, keyed
by rotary_scale and token count.
MEASURED END TO END (tttLRM from-scratch grid, 300 s window, 2 GPUs/cell):
| arm | before | after | vs base |
|---|---|---|---|
| base | 2.72 | 2.50 | . |
| input | 2.93 | **2.50** | **0%** |
| hidden | 3.62 | 2.73 | +9% |
| **Both** | 3.85 | **2.73** | **+9%** |
The input site's overhead is now zero. `base` also gained 8%, because the global switch
had been suppressing unrelated compiles (`silu_backprop` and friends).
Applied to all four tasks: tttLRM, NVS, video/CCV, LLM.
**REPRODUCIBILITY CAVEAT, and it is real:** the fused kernel is NOT bit-identical to
eager. Inductor reorders the rounding, giving 1 ULP in bf16 (9.77e-04, up to 0.0156 at
full ladder width; 9.5e-07 in fp32). Comparisons WITHIN an experiment stay exact in kind,
since all arms share the code path, and 1 ULP is orders of magnitude below our own noise
floor (F18: 0.1-0.3 dB PSNR gaps are seed noise). But numbers produced after this change
do not bit-reproduce numbers produced before it, which includes F25, F40 and F41.
`TTTROPE_NO_COMPILE=1` restores the eager path if an exact re-run is ever needed.

## F44 (was mis-numbered F41; the n-step finding below owns F41): Q31 attn_nope — removing the ATTENTION rotary FLIPS the hidden rotary from
## harmful to helpful, and leaves the input rotary's gain untouched (2026-08-05)
200M LaCT LM, 3B tokens fineweb-edu, data_seed 42, bs 8x4096, window 1024, 91,552
steps — F27's protocol exactly, with `attn_nope=true` in every cell so the
sliding-window attention carries no explicit positional code and the TTT rotary is
the model's only one. Wiring guarded before launch: the flag reaches 12/12 layers
and changes the forward (|dloss| = 1.5e-05 untrained / 2.7e-04 trained, matched
state_dict) — a silent no-op would have been indistinguishable from a null (cf. Q29).

VAL-CACHE FIX. Q31 logged against `val_cache_..._ds42.pt`; F27 used
`val_cache_..._4096.pt`, since deleted. As logged the two columns are NOT
comparable. The F27-era cache was restored from git and the Q31 checkpoints
re-scored on it, so the table below is a SAME-SAMPLE comparison (`paired_lm.py`,
raw in `outputs/q31_paired_F27cache.json`).

| Δ ppl vs its own nope | attn rope ON (F27) | attn rope OFF (Q31) |
|---|---|---|
| input fw-RoPE (`in`) | −0.22 | **−0.209** |
| hidden rotary (`h`) | +0.23 | **−0.112** |
| both | +0.02 | **−0.175** |

Q31 absolute (F27-era cache): nope 20.558, in 20.349, h 20.447, both 20.383.
Paired per-block over n=488, vs `nope`: in −0.0102 nats (t=−18.08, win 79.5%),
h −0.0054 (t=−9.93, 69.7%), both −0.0086 (t=−15.65, 75.8%).

- THE HYPOTHESIS SPLITS BY SITE. "Attention already supplies position, so the
  fast-weight rotary has nothing left to contribute" is SUPPORTED for the hidden
  site (+0.23 → −0.11: it stops hurting and starts helping) and REFUTED for the
  input site (−0.22 → −0.21: unchanged, so its gain was never redundancy with
  attention). Whatever the input rotary buys, attention was not already providing.
- The arms do NOT collapse together; they order in→both→h→nope with every rotary
  arm now beating nope.
- Absolute ppl worsens 18.62 → 20.56 on nope, as pre-registered. We removed a
  useful code. This is a channel-value decomposition, NOT a SOTA claim.
- SCOPE: causal masking still leaks position, so this is "no explicit positional
  code", not "position-free" — that leak is why NoPE transformers work at all
  (Kazemnejad et al.).
- CONFOUND, NOT YET CLOSED: the ON column is F27 (2026-07-09) and the OFF column
  is Q31 (2026-08-05), i.e. different environment instances. F27b established that
  env changes alone shift absolute ppl ~0.9 and FLIPPED this very contrast once
  (hidden −0.19 old-env vs +0.24 new-env). The observed flip (+0.23 → −0.11) is
  the same magnitude as that env-induced flip, so it CANNOT yet be attributed to
  attn_nope. Closing it requires re-running the attn-rope-ON column in the current
  env (4 cells, ~5 h each). Until then this is a strong lead, not an established
  result.
- Seed replication of the OFF column (seed 43, data_seed held at 42) is in flight;
  within-column paired t of −9.9..−18.1 is block-level noise only, not init noise.

## F41: TTT-RoPE survives the n-step update, and its value GROWS with it (2026-08-05)
The method is derived for a SINGLE update step: the phases cancel inside one inner
product, one gradient step. At scale the update is split into n sequential chunks, each
updating the fast weights on top of the previous chunk's result, and the derivation does
not cover that. tttLRM already runs this way (`full_ttt_op`, update_minibatch 1024)
while NVS does one update over all input tokens, so this is also a live candidate for
the NVS-vs-3D-reconstruction difference.
EVALUATION ONLY: the same 30k/seed-95 checkpoints, all TRAINED with a single update, are
re-evaluated with the input-token update split n ways. 32 input views = 8192 update
tokens, so chunk size is 8192/n and every setting stays above Muon's ~427-token
amortisation point. 256 held-out scenes, paired per scene.
| arm | n=1 | n=2 | n=4 | n=8 |
|---|---|---|---|---|
| NoPE | 21.931 | 19.646 | **15.936** | **14.303** |
| input | 22.726 | 22.049 | 20.393 | 18.358 |
| hidden | 23.057 | 21.583 | 17.312 | 15.251 |
| **Both** | **23.365** | **22.701** | **20.961** | **19.067** |
Paired delta vs NoPE (t in parentheses):
| arm | n=1 | n=2 | n=4 | n=8 |
|---|---|---|---|---|
| input | +0.795 (+36) | +2.404 (+61) | +4.457 (+62) | +4.054 (+51) |
| hidden | +1.126 (+50) | +1.937 (+53) | +1.375 (+37) | +0.948 (+21) |
| **Both** | +1.433 (+54) | +3.055 (+68) | **+5.025 (+69)** | +4.763 (+68) |
Readings:
1. **The rotary's value more than triples under chunked updates** (Both +1.43 -> +5.03)
   and stays significant at every n. `Both` leads at every n. The extension holds.
2. **The mechanism is visible in the NoPE row: chunking COLLAPSES the baseline**
   (21.93 -> 14.30). In a sequential update each chunk writes on top of the previous
   fast weights and per-chunk weight-norm decays the earlier ones, so without an address
   space the earlier chunks are overwritten. With a rotary the chunks sit at different
   phases and stay separable. Sequential updating is therefore the regime where
   addressing matters MOST, not a regime the method merely tolerates.
3. The two sites respond OPPOSITELY: input's delta grows with n (+0.80 -> +4.05) while
   hidden's shrinks (+1.13 -> +0.95). Worth stating; not yet explained.
4. This rules chunking OUT as the explanation for `Both` trailing the single-site arms
   on 3D reconstruction: here more chunks make `Both` MORE favoured. The remaining
   candidate is ladder width (3D recon rotated 8.2% of the hidden vs NVS's 98.4%), which
   the nf 64/256 re-run tests directly.
SCOPE: these checkpoints were trained with one update, so this shows the learned phases
survive chunked application. It is NOT the same as showing the method trains well in the
n-step regime; that is the multi-chunk training run (queued, `ttt_num_chunks` accepts a
list and draws one n per forward). n=16 was measured too and continues the pattern
(Both +3.281) but is left out of the table.

## F43 (was mis-numbered F40; the NVS view sweep below owns F40): Q29 CLRS-Text address-DIMENSION grid — the 2-D address helps BOTH sites
## equally; the hidden increment does NOT grow with dimensionality (2026-08-05)
CLRS-Text 2d_long, 12L/768d/4 lact heads, seq 4096, window 128, 1.2B tokens,
seed 42, held-out CLRS test seeds. `coord_mode=1d` feeds (t, t), which recombines
the split ladder into inv_freq*t — bit-identically the stock rotary — so 2d-vs-1d
moves the address dimension and NOTHING else (same data, tokens, length, load).
Paired per-problem over n=488 held-out blocks (pooled answer_acc in parens):

| contrast | mean d | t | win% |
|---|---|---|---|
| in_1d − base | −0.0346 | −10.38 | 24.6 |
| h_1d − base | −0.0289 | −11.51 | 19.3 |
| both_1d − base | −0.0350 | −14.79 | 6.1 |
| in_2d − base | −0.0156 | −15.27 | 15.0 |
| h_2d − base | −0.0103 | −4.62 | 46.3 |
| both_2d − base | −0.0166 | −7.84 | 36.7 |
| **h_2d − in_2d** | **+0.0053** | 2.26 | 66.6 |
| **h_1d − in_1d** | **+0.0057** | 4.24 | 41.6 |
| in_2d − in_1d | +0.0189 | 5.76 | 35.5 |
| h_2d − h_1d | +0.0185 | 10.47 | 70.1 |

pooled: base .8559, h_2d .8414, in_2d .8407, both_2d .8350, h_1d .8240,
both_1d .8182, in_1d .8164.

- PRE-REGISTERED PREDICTION REFUTED. The prediction was that the h-over-in
  increment GROWS in the 2-D arms and not in the 1-D arms. It is FLAT: +0.0057
  (1d) vs +0.0053 (2d). Whatever the hidden site buys on this task, a
  higher-dimensional address is not what unlocks it. F20's dimensionality
  hypothesis is refuted FOR CLRS.
- The 2-D address is nonetheless strongly load-bearing, and helps both sites by
  the SAME amount (in +0.0189, h +0.0185). So dimensionality is real but
  site-agnostic.
- NoPE BEATS EVERY ROTARY ARM (t = −4.6 .. −15.3). On algorithmic traces the
  retrieval is content-keyed, and a positional phase on Q/K stops equal-content
  tokens at different positions from matching — pure nuisance the model must
  cancel. The 2-D address halves the deficit (−0.032 → −0.011) but never
  overturns it. This is the OTHER SIDE of the inner-product addressing lemma:
  the rotary pays off when the retrieval key IS relative geometry (NVS), and
  costs when it is content.
- SEED CAVEAT: the t-statistics are paired over BLOCKS and control block
  difficulty, not initialisation. One seed per arm. The base-vs-rotary gaps and
  the +0.019 dimensionality gain are far too large to be seed artifacts; the
  h−in +0.005 (t=2.26) is NOT — it is within what a seed swap produces (cf.
  F33/F18). "Hidden beats input" remains UNESTABLISHED; 2-3 seeds per arm would
  be needed.
- VALIDITY: the first Q29 grid returned bit-identical 2d/1d numbers because
  `ttt_layers` was empty, so the address never reached the rotary. Those runs are
  quarantined as `outputs/q29_*_INVALID_stockrotary`. Every cell here carries the
  startup guard line (`COORD VERIFIED ACTIVE`, |d| = 1.1e-01 / 1.4e-01 / 3.5e-02;
  base `address correctly inert`, |d| = 0.000e+00).
- Stats: `lact_llm/paired_clrs.py`, raw in `lact_llm/outputs/q29_paired.json`.
- SCOPE: procedurally-generated algorithmic traces, not natural language. The
  3-seed natural-language LM null (F33) stands unchanged.

## F40: Figure 1 NVS view sweep — the rotary's value GROWS with input scale, and the
## baseline SATURATES (2026-08-05)
Evaluation-only sweep: the SAME 30k/seed-95 checkpoints re-evaluated at 4..32 input
views, 256 held-out scenes, paired per-scene. Models were trained at 8 input views, so
16/24/32 are extrapolation. n=2 excluded (two views barely constitute a reconstruction).
| views | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| NoPE | 20.647 | 21.825 | **22.031** | 21.937 | 21.931 |
| input | 20.661 | 22.333 | 22.766 | 22.677 | 22.726 |
| hidden | 21.112 | 22.724 | 23.087 | 23.024 | 23.057 |
| **Both** | 20.886 | 22.797 | 23.341 | 23.279 | **23.365** |
Paired delta vs NoPE (t in parentheses):
| arm | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| input | +0.013 (+1) | +0.508 (+21) | +0.735 (+33) | +0.740 (+32) | +0.795 (+36) |
| hidden | +0.464 (+18) | +0.899 (+41) | +1.057 (+50) | +1.087 (+46) | +1.126 (+50) |
| Both | +0.239 (+7) | +0.971 (+32) | +1.310 (+49) | +1.342 (+48) | **+1.433 (+54)** |
Readings:
1. **THE HEADLINE: NoPE saturates, the rotary arms do not.** NoPE peaks at 16 views
   (22.031) and then goes flat/down (21.937, 21.931), while Both keeps climbing to
   23.365. The fixed-size fast weights cannot address more content; the rotary gives
   them an address space and the gain rises monotonically 8 -> 32 views
   (+0.97 -> +1.43).
2. This was PRE-REGISTERED before the numbers existed (`EXPERIMENT_QUEUE_PAPER.md`,
   `paper_overleaf/experiment.md`): "if the rotary works by giving the fast weights a
   usable address space, its advantage should grow with input size... a flat gap would
   weaken the thesis". It grew.
3. **The gain grows in EXTRAPOLATION.** Training used 8 views; the largest deltas are
   at 16-32. So this is a robustness property, not a fit to the training regime.
4. Ordering is stable across the whole sweep: Both > hidden > input > NoPE, from 8
   views on. At 4 views input is worth nothing (+0.013, t=+1) while hidden already
   earns (+0.464) - the hidden site starts paying earlier.
5. Boundary at the small end: at 2 views (excluded from the table) the rotary HURTS
   (Both -0.309, input -0.198). With almost nothing to address, rotation is pure cost.
Figure: `lact_nvs/fig1_nvs_viewsweep.png`. Data: `lact_nvs/fig1_viewsweep/*.json`
(per-scene arrays kept, so any panel can carry paired error bars).
The NoPE checkpoint did not survive the node reset and was retrained on node2 as P1.

## F39: Q28 tttLRM warmup grid, step-1000 held-out — the fine-tune design is a
## STRUCTURAL dead end, not a budget shortfall (2026-08-04)
DL3DV-140, 140 scenes, 32 input views, K-means selection, 536x960 — the same
protocol that reproduced their Table 2 (our anchor 25.062 vs published 25.07).
Cells fine-tuned from the released checkpoint with a 200-step gain ramp.
| cell | PSNR | SSIM | LPIPS | dPSNR vs base (paired) | t | improved |
|---|---|---|---|---|---|---|
| base | **25.050** | 0.8178 | 0.2176 | — | — | — |
| in (qk_rope_cam) | 23.733 | 0.7759 | 0.2593 | -1.317 +- 0.039 | -33.9 | 0/140 |
| h (h_pra) | 24.807 | 0.8101 | 0.2250 | **-0.243 +- 0.010** | -24.7 | 0/140 |
| both | 23.625 | 0.7727 | 0.2633 | -1.425 +- 0.040 | -35.4 | 0/140 |
Readings:
1. **THE DECISIVE ONE — base 25.050 vs anchor 25.062: fine-tuning does not move
   the baseline at all.** The released checkpoint is already converged on DL3DV,
   so this design has ZERO headroom: the only outcome it can ever produce is
   "the rotary costs something". That is a structural property of grafting onto
   a converged representation, NOT a consequence of the 2000-step budget, and it
   is why more compute would not have rescued it. Any claim about whether the
   rotary HELPS requires training from scratch, where all cells start equal.
2. F30's train/held-out divergence did NOT recur. It was the stated reason to
   check held-out before abandoning (F30: training-neutral -> held-out -8.7%),
   but here held-out agrees with training in both sign and ordering
   (h training delta -0.19..-0.21 vs held-out -0.243).
3. **The hidden site is ~5.4x cheaper to graft than the input site**
   (-0.243 vs -1.317), now confirmed on 140 held-out scenes at t=-24.7, not just
   on training loss. `both` (-1.425) ~= `in` (-1.317): the damage comes almost
   entirely from the input site, and adding the hidden rotary on top costs a
   further 0.11 dB. Fifth independent measurement of this ratio.
   Publishable as its own finding: the hidden rotary transplants nearly free onto
   a concat-trained pretrained spatial memory; the input rotary does not.
4. The warmup ramp halved the shock vs the no-warmup grid (-2.3 -> -1.0 dB in
   training) but did not change the sign or the plateau.
NEXT: from-scratch at half scale (272x480, 8+8 views, d768/24L unchanged),
2 GPUs x 12h per cell, wave 1 = base + in. See EXPERIMENT_QUEUE.md Q28.

## F38: depth-4 fast weights — the rotary-depth interaction SATURATES at depth 3 (2026-08-03)
Motivated by the F24/F26 reading that "deep inner models don't help" is an ADDRESSING failure: if a
rotary per address space is what makes depth pay, does depth 4 keep paying? New kernel
`fast_weight_swiglu4l_weight_norm_apply` + cam_modes `fw4l` / `fw4l_rot4` (4 rotary sites: input q/k
+ 3 hidden interfaces). Standard protocol, seed 95; base-vs-rot4 is param-neutral (+5,292 rotary gains).
| depth | no rotary | with rotary | rotary's value |
|---|---|---|---|
| 2L | 21.745 | 22.824 | +1.08 |
| 3L | 21.868 | 23.439 (3-seed +-0.022) | **+1.57** |
| 4L | **21.896** | **23.410** | **+1.51** |
- fw4l_rot4 - fw4l_base = **+1.513 dB, t=+27.8, 251/256 scenes** (s95; s137 running).
- THE NO-ROTARY AXIS IS FLAT ACROSS THREE DEPTHS (21.745 / 21.868 / 21.896): capacity alone buys
  nothing, now confirmed at a third point. The "addressing, not depth" claim is stronger than before.
- BUT the interaction SATURATES: 2->3 raised the rotary's value (+1.08 -> +1.57) while 3->4 did not
  (+1.57 -> +1.51), and the absolute record does not move (23.439 -> 23.410, within the 3-seed band).
  Honest statement for the paper: one rotary per address space pays until the addressing capacity
  matches the task's geometry; at L6/d256 on RE10K that point is depth 3. Do NOT frame depth as a
  scaling axis.
- Sanity for the new kernel: zero-phase reduction to fw4l_base exact (fp32 max|diff| 0), manual
  inner-loop backward vs autograd 1.7e-18, early training curve indistinguishable from fw3l_rot3,
  ~10% slower per iteration. Depth costs +1.57M params (3L->4L), so the depth axis is NOT param-neutral
  and must be reported as such; the base-vs-rot contrast within each depth IS param-neutral.

## F38: depth x rotary interaction SATURATES at depth 3 (2026-08-03)
Added the depth-4 point (new `fw4l` / `fw4l_rot4` cam_modes: 4-layer inner net, one rotary per
address space = input + 3 hidden interfaces; base-vs-rot4 is param-neutral, +5,292 rotary gains).
| fast-weight depth | no rotary | with rotary | rotary's value |
|---|---|---|---|
| 2L | 21.745 | 22.824 (3-seed) | +1.08 |
| 3L | 21.868 | 23.439 +- 0.022 (3-seed) | +1.57 |
| 4L | 21.896 (s95) | 23.410 (s95) | **+1.51** |
fw4l_rot4 vs fw4l_base paired: **+1.513 dB, t=+27.78, 251/256 scenes**.
Readings:
1. **Depth alone stays worthless all the way to 4L**: 21.745 -> 21.868 -> 21.896 (+0.15 total
   while doubling depth). The capacity axis is inert in this architecture.
2. **Rotary's value grows 2L->3L (+1.08 -> +1.57) but PLATEAUS at 4L (+1.51)**, and the absolute
   best is a tie between 3L-rot3 (23.439, 3-seed) and 4L-rot4 (23.410, 1 seed) — while 4L costs
   +1,574,406 params over 3L. Correct claim: the depth-rotary interaction saturates at depth 3;
   TTT-RoPE does not unlock unbounded depth.
3. The headline stands and is now bracketed on both sides: without rotary, added address spaces
   are dead weight; with rotary, the same architecture earns +1.5..1.6 dB. The "deep inner models
   don't help" folklore is an ADDRESSING failure, not a depth failure — but the fix has a ceiling.
Caveat: the 4L cells are single-seed (s95); the 3L-vs-4L gap (-0.03) is inside seed noise, so
"tie" is the claim, not "4L is worse". Impl: lact_ttt_cam.py fast_weight_swiglu4l_weight_norm_apply
(hand-derived depth-4 backward, verified vs autograd to 1.7e-18; zero-phase reduction exact).

## F37: Q13 NVS — SHARING the learnable ladder across layers does not rescue it (2026-07-17)
Hypothesis (user): per-layer learnable gains lose to fixed because 6 layers x 6xF gains
is too much freedom; ONE shared gain tensor might fix it. Paired wave-1 (seed 95,
standard protocol, 'sharedf' registry so the optimizer sees the parameter once):
| variant | PSNR | LPIPS |
|---|---|---|
| qk_rope per-layer learnable (fresh rerun) | 22.420 | 0.2769 |
| qk_rope sharedf (one ladder, all layers) | 22.338 | 0.2782 |
shared - per-layer = **-0.082 dB, t=-12.52** (shared better in only 56/256 scenes).
Sharing strictly HURTS: the freedom-reduction hypothesis is refuted in NVS (and its
LLM analogue died independently as F33's shared-ladder seed lottery). Combined with
Q20 (per-head) and Q21 (LieRE b2/b8, both inits, honly+hpra — all 4 cells at w128
worse than fixed by +0.07..+0.22), every granularity of learnable frequency ladder
(shared / per-layer / per-head / joint planes+angles) now has a negative result on
at least one task. Fixed ladders stay the paper recipe. No Q13 wave 2.

## F36: Q17 window sweep — the hidden rotary's value GROWS with memory workload to PARITY with the input rotary (FINAL, 3 seeds; the s42 "overtake" did not survive s211) (2026-07-16, seeds 2026-07-17)
Lever: shrink the sliding-window attention 1024 -> 128 so the fast-weight memory becomes
load-bearing for the mid-range positional structure in NATURAL language (memory-exclusive
positions 31.5% -> ~90%). Full 6-cell grid, 3B ds42, standard protocol.
Channel value = ppl reduction vs NoPE (the clean decomposition):
| rotary | window 1024 | window 128 |
|---|---|---|
| input (rope) | +0.22 | +0.20 |
| hidden (honly) | +0.09 (g0.1) | **+0.26 (g1.0)** |
Absolute (w128): nope 18.81 / rope 18.61 / honly-g1.0 **18.55** / hpra 18.64 (both ladders).
Readings (all s42, single seed — REPLICATION s137/s211 RUNNING):
1. **The input rotary's value is window-INVARIANT (~0.2); the hidden rotary's value
   TRIPLES as the window shrinks and OVERTAKES the input** (0.26 > 0.20). The NoPE anchor
   proves this is the hidden channel getting more valuable, not the input getting worse.
   First quantitative natural-language evidence for the load-bearing-memory picture that
   NVS/ccv/copy showed: the hidden site lives in the memory's address space, so its value
   scales with how much work the memory does.
2. **hidden-only BEATS input-only at w128** (18.55 < 18.61) — reverses the w1024 ordering
   (there honly 18.53 > rope 18.40). In a load-bearing regime the best single place for
   the rotary is the HIDDEN site.
3. Ladder-band rule confirmed again (F35): at w128 the STANDARD ladder wins for honly
   (g1.0 18.55; g0.1 pending) — the gentle-ladder rule was a w1024-band artifact.
4. ODDITY: combining both (hpra 18.64) is WORSE than either alone (rope 18.61, honly
   18.55) — destructive when stacked at w128. So this is hidden REPLACING input, not
   input+hidden > input; the original goal cell (hpra > rope) still fails.
Seed replication FINAL (3 seeds): s42 -0.06, s137 -0.04, **s211 +0.13** (honly 18.67
vs rope 18.54) — the REVERSAL DOES NOT SURVIVE the third seed (mean gap +0.01, mixed
signs). Corrected 3-seed statement: at w128 hidden-only becomes statistically
INDISTINGUISHABLE from input-only (up from consistently ~+0.13 behind at w1024). The
load-bearing TREND is real — the honly-rope gap shrinks from ~+0.12 (w1024) to ~0
(w128), consistent with the channel-value decomposition — but "hidden beats input" in
natural language is NOT established. The decomposition claim (hidden value grows with
memory workload) stands as a trend; the overtake was seed luck at s42/s137.

## F35: Q16 exact-offset copy — the 1D hidden rotary CARRIES precise positional retrieval when the task demands it (2026-07-16)
Task: 256-token random span, reproduce at offset 2560 (= 2.5x the 1024 attention window,
crosses 2 chunk boundaries) — only the fast-weight memory can carry it; the answer
depends on exact relative offset. 200M production arch, 800M tokens, copy-region loss,
grokking-like transition. Grid (final copy accuracy | transition step acc>50%):
| variant | final acc | transition |
|---|---|---|
| NoPE | **0.2% (random floor — NEVER learns)** | never |
| rope (input) | 100% | 8500 |
| honly g0.1 | 100% | 11000 |
| **honly g1.0** | 100% | **4000 (fastest)** |
| hpra g0.1 | 100% (loss 0.0002, lowest) | 6000 |
| hpra g1.0 | 100% | 4500 |
Readings:
1. WITHOUT a positional code the task is unlearnable at this budget (content-induction
   alternative did not materialize) — position is effectively necessary.
2. **The hidden rotary ALONE fully solves precise long-range retrieval** (honly 100% vs
   NoPE 0.2%): first clean demonstration that the 1D hidden site can carry exact
   positional addressing when the memory is load-bearing and the task rewards precision.
   The 1D failures in natural language (F27/F33) were the TASK's thinness, not an
   inability of the hidden channel.
3. Ladder-band prediction confirmed: on a precision task the STANDARD ladder beats the
   gentle one (honly g1.0 transitions at 4000 vs g0.1 at 11000) — the F28 gentle-ladder
   rule was a property of the w1024 natural-language band, not universal.
4. Stacking accelerates: hpra transitions before rope at both gains (4500/6000 vs 8500)
   and reaches the lowest loss — on THIS task input+hidden > input in learning speed,
   though all saturate at 100%.
Contrast cell (same day): Q17-A w128 natural language — tripling the memory-exclusive
zone did NOT open the hpra-rope increment (18.61 vs 18.64). Together: the binding
constraint in natural language is that its long-range retrievals are content-addressed,
not position-addressed; when retrieval is position-addressed (copy), every rotary site
earns and hidden alone suffices. Remaining w128 cells (nope/honly, g1.0) will refine.

## F34: Q15 — faithful PRoPE port WINS; its gain is the orthogonal component (2026-07-15)
Trigger: a coworker reports LaCT + PRoPE (as-is) beats baseline. Our F3 cell was a
LOSS (-0.118) — but our old port re-L2-normalized q/k AFTER the projective transform
(breaking the score cancellation), tiled only half the head dim, and omitted PRoPE's
image-coordinate ropes. Faithful port from the official reference (prope/prope/torch.py):
q/k/v/o each get [head_dim/2 tiled projective P | head_dim/4 image-x RoPE | head_dim/4
image-y RoPE], freq_base 100, split pairing, inverse rotations on o; applied after our
fast-q/k L2-norm. Grid (standard protocol, s95, baseline 21.970 / our input rotary
22.375 / our full recipe 22.971 3-seed):
| cell | PSNR | delta |
|---|---|---|
| **prope_orig (faithful)** | **22.255 / LPIPS 0.2795** | **+0.285 — coworker REPLICATED** |
| prope_ttt (old F3 port) | 21.852 | -0.118 |
| gta_in (rigid, q/k, pre-norm renorm) | 21.833 | -0.137 |
| prope_in (projective, q/k, pre-norm renorm) | 21.786 | -0.184 |
| prope_raw (= prope_orig MINUS image ropes) | 21.676 | -0.294 |
| **prope_imgrope (= prope_orig MINUS projective)** | **22.349 / LPIPS 0.2767** | **+0.379** |
Readings:
1. The coworker's claim replicates in our stack once the port is faithful.
2. The DECOMPOSITION vindicates F1/F3's mechanism and is now airtight: the orthogonal
   image-rope component ALONE scores +0.379 (nearly our input rotary's +0.405); adding
   the projective half SUBTRACTS ~0.09 (+0.379 -> +0.285); the projective half alone
   is -0.294. PRoPE's entire gain in the LaCT stack is its orthogonal rotary part;
   the projective transform is a consistent liability in every arrangement (6 cells).
3. Ranking preserved: faithful PRoPE (+0.29) < our input rotary (+0.41) < our full
   recipe (+1.08; fw3l_rot3 +1.69). F3's claim needs scoping, not retraction: the
   projective TRANSFORM is what loses; PRoPE-the-package wins via its rotary part.
s137 replication (2026-07-16): prope_orig 22.019, prope_imgrope 22.110 — both above
the (3-seed-mean) baseline; the projective penalty is eerily seed-stable (imgrope minus
orig: +0.094 s95, +0.091 s137); ordering vs our input rotary holds per seed
(imgrope 22.110 < pra_hi_s137 22.385; s95: 22.349 < 22.375). Two-seed verdict FINAL:
faithful PRoPE works, its engine is the orthogonal part, and our rotary stays ahead.
Paper untouched (freeze); this entry is the record.

## F33: Q12 stacking program CLOSED — no 1D hidden increment survives seeds; learnable ladders are an init lottery (2026-07-15)
Program (user /goal): make rope+hidden < rope (18.405/18.19/18.26 at s42/137/211) at
the F27 3B protocol. 16 stacked variants + seed replication, all on identical data
(ds42) with per-step seeded draws. Complete inventory:
1. FIXED gentle ladder (champion g0.1): +0.01 (s42) / +0.19 (s137) — non-negative
   both seeds; the F28-addendum s42 "neutral" was the favorable draw.
2. Gain/frac/theta axes (0.05/0.15/0.2, frac 25/75): all +0.05..+0.29 at s42.
3. Mechanism axes: delta-only +0.22, hnorm-rms_rot +0.71, late-layer(8-11) +0.10,
   per-layer learnable +0.23, learnable input deltas +0.33 (harmful even from
   ZERO init: sharedHI0 18.42 vs sharedH 18.20 at s42).
4. SHARED learnable hidden ladder (Q13 idea): the one s42 winner — 18.204 (-0.200,
   gap smooth from 7k; init-robust at s42: g1.0-init 18.17). But 3-seed kills it:
   s137 18.75 (+0.56), s211 18.92 (+0.66). Deterministic init (tilt=0) does NOT
   rescue s137 (18.84): the fragility is the model-init x learnable-ladder training
   dynamics, not the tilt draw. Gap sign is set in the first ~4k steps and stays
   smooth — a mirror of the s42 win. Extends F20/F29: 1D learnable frequency
   ladders are an initialization lottery at this scale, in every parameterization
   (per-layer, shared, deterministic-init, input-additive).
VERDICT: in 1D at 200M/3B, the hidden site does not add on top of the input rotary.
What stands: hidden-only gentle ladder beats NoPE (18.53/18.64 2-seed, F28) — the
1D hidden channel is real but subsumed by the input score. The graded-boundary
narrative is now backed by 3 domains x 16 variants x 3 seeds. Remaining (untested,
out of protocol): longer contexts / larger models, where 1D relative structure and
memory load both grow. Ledger: lact_llm/ga_honly/LEDGER.md (Q12 waves 1-6).

## F32: Q11-S1 — the hidden increment APPEARS in the frozen regime once the memory works (2026-07-14)
Stage-1 recipe on the frozen-ReCamMaster adapter (user-approved): fast-weight capacity
x2 (inter_multi 4), 21x1-frame update chunks, Muon on the chunk updates, ReCamMaster's
own cam_encoder+projector trainable (1e-5 group; Wan still fully frozen), 6000-pair
index, 6000 steps (1 epoch). Same eval protocol as F31.

Val loss (64 pairs x t{100,500,900}, paired n=192):
| variant | mean | vs base | t |
|---|---|---|---|
| base_s1 | 0.12253 | — | — |
| in_s1 | 0.11772 | -3.93% | -6.53 |
| h_s1 | 0.12078 | -1.43% | -3.95 |
| both_s1 | **0.11620** | **-5.16%** | **-7.33** |
**both-in (hidden increment) = -1.24%, t=-6.08 (130/192)** — zero at F31 (3250 steps,
weaker memory recipe: +0.06%, n.s.), decisive after Stage-1. The increment tracks how
hard the memory works, in a SECOND video regime (frozen backbone) after F30
(full-training, t=-9.0). Rotary deltas grew ~5x vs F31 (in: -0.72% -> -3.93%).

Generation (8 pairs, official sampler): base 12.58/0.4234/0.6289 ->
in 14.55/0.4774/0.5142, h 14.45/0.4597/0.5301 (8/8 pairs LPIPS), both
14.49/0.4711/**0.5092** (PSNR/SSIM/LPIPS). Rotary buys ~2 dB in generation; the
rotary variants' LPIPS now BEAT the old full-replacement ccv best (0.545 @14k steps)
with a frozen backbone and 6000 adapter steps. both-in n.s. at n=8 (t=-0.69); val loss
is the discriminator. Gap to ReCamMaster (15.71/0.453): base recovery 42% of the
removed-channel hole (19% at F31); in/both ~49%.
Mechanics kept honest: all Stage-1 changes shared by all four variants; identical
data stream + per-step seeded noise; sanity 16/16 incl. muon-off bitwise-equal to the
old kernel (commit f2ea68b).

### F32b: 12k-step extension of in/both (fresh cosine, 2026-07-16)
Doubling the schedule (6000 -> 12000 steps, fresh cosine) sharpens everything:
| variant | 6k val | 12k val | 12k gen (PSNR/LPIPS) |
|---|---|---|---|
| in_s1 | 0.11772 | 0.09951 | 16.11 / 0.4282 |
| both_s1 | 0.11620 | **0.09706** | **16.34 / 0.4152** |
- Both improve hugely over 6k (in -0.0182 t=-12.6; both -0.0191 t=-13.8) — the 6k probe
  was NOT plateaued, memory recall keeps improving with budget.
- **Hidden increment SHARPENS: both-in = -0.00245, t=-10.05 (178/192 pairs)** vs t=-6.08
  at 6k. The more the memory works (longer training), the more the hidden rotary earns —
  cleanest confirmation yet of the load-bearing-memory picture.
- **Generation now BEATS the ReCamMaster anchor**: both 16.34 dB / LPIPS 0.4152 vs
  ReCamMaster 15.71 / 0.4534. A frozen-backbone TTT adapter (12k steps, ~1 B200-day)
  surpasses the released 8xH800x3day model on our held-out pairs, and the input+hidden
  rotary is the best cell on both metrics.

**COMPLETE 12k 2x2 (base/h added 2026-07-17)** — val loss (n=192 paired) and
generation (8 pairs, official sampler):
| variant | 12k val | vs base | t | gen PSNR / SSIM / LPIPS |
|---|---|---|---|---|
| base | 0.11390 | — | — | 14.27 / 0.464 / 0.552 |
| h    | 0.10893 | -4.36% | -7.87 (180/192) | 15.16 / 0.487 / 0.477 |
| in   | 0.09951 | -12.63% | -11.08 (191/192) | 16.11 / 0.524 / 0.428 |
| both | **0.09706** | **-14.78%** | -12.17 (192/192) | **16.34 / 0.533 / 0.415** |
Increments: both-in -2.47% t=-10.05 (178/192); both-h -10.90% t=-12.22; every cell
significant, and the ordering both > in > h > base is UNANIMOUS across val loss,
PSNR, SSIM, and LPIPS. Both sites earn independently on a frozen task-tuned
backbone (h alone -4.4% over base at t=-7.9 — the hidden channel works standalone
here, unlike F31's 3250-step readout), and stacking beats each single site. both
and in both beat the released ReCamMaster anchor (15.71/0.4534); h alone does not.
This is the cleanest full-hierarchy confirmation of "one rotary per address space"
outside NVS.

## F31: Q11 frozen-ReCamMaster + TTT-adapter 2x2 (fixed ladders, 3250 steps, 2026-07-12)
Design (user pivot): released ReCamMaster step20000, EVERYTHING pretrained frozen
(incl. their fine-tuned self_attn + cam_encoder); attention reverted to per-video;
a zero-init TTT fast-weight branch (496M params, update on 7 src chunks / apply on
tgt) is the ONLY cross-video channel. 4 runs (base/in/h/both), fixed Plucker
ladders, identical data stream + per-step seeded noise/timesteps, lr 1e-4 cosine,
bs1, 3250 steps (~6h/GPU). Trainers in lact_ar_video/recam_ttt/ (commits e29f6bc,
11f4c6b, 88bf9e7).

Phase-1 val loss (64 pairs x fixed timesteps {100,500,900}, paired n=192):
| variant | mean loss | delta vs base | t | lower |
|---|---|---|---|---|
| base | 0.140407 | — | — | — |
| in   | 0.139392 | **-0.72%** | **-6.15** | 156/192 |
| h    | 0.140045 | -0.26% | -3.05 | 143/192 |
| both | 0.139481 | -0.66% | -5.48 | 143/192 |
both-in = +0.06% (t=+0.95, n.s.); in-h = -0.47% (t=-4.86).

Phase-2 generation (official sampler 50 steps CFG 5, 8 pairs): base 10.74/0.368/
0.718 (PSNR/SSIM/LPIPS), in 10.93/-/0.716, h 10.46/-/0.730, both 11.08/0.374/0.715.
Deltas n.s. at n=8 (in/both directionally positive: PSNR +0.19/+0.34, t=+1.7).

Reading:
1. THE ROTARY EARNS EVEN ON A FROZEN, TASK-TUNED BACKBONE: input site -0.72% at
   t=-6.15 with only 3250 adapter steps, on top of an existing camera channel
   (frozen cam_encoder). This is the cleanest "one knob, everything else frozen"
   isolation of PRA so far.
2. SITE HIERARCHY FLIPS vs the full-replacement ccv grid (F30: hidden increment
   t=-9.0): here INPUT dominates and hidden adds nothing over it (t=+0.95). With
   the fast-weight branch as a fresh adapter doing content transport next to a
   frozen absolute-pose channel, the addressing-level relative geometry is
   captured by the input rotary alone; the hidden site's extra leverage appears
   when the whole layer stack is trained around it (F30) — consistent with the
   graded-boundary story (F21/F22/F28).
3. ABSOLUTE QUALITY GAP: all four sit at PSNR ~10.5-11.1 / LPIPS ~0.72 vs
   ReCamMaster's 15.71/0.453 (F30c) — 3250 steps of a zero-init adapter do not
   yet replace the concat-attention channel their 20k-step 8xH800 fine-tune
   built. The 2x2 delta is paper-usable; the absolute row needs longer training
   (fresh longer cosine, not a post-anneal extension) if wanted.
Artifacts: outputs/recam_ttt/{valloss,gen}_*_3250*, training logs + jsonl with
probe_val trajectories; eval protocol row-compatible with F30c's external anchor.

## F30: ccv held-out eval — the video boundary FLIPS when memory is load-bearing (2026-07-12)
New eval path (commit 92e2486; Phase-1 = deterministic held-out val loss, 64 fixed pairs
disjoint from the training index, per-pair fixed noise/timesteps; EMA weights, common
checkpoint step 13999):
**LABELS CORRECTED 2026-08-05.** The original table called these arms "input",
"input, fixed" and "input+hidden". That is WRONG, verified against the config each run
actually saved (`outputs/ccv_*/seed_1/config.yaml`), not just the repo configs. The
real axis is **camera-as-features vs camera-as-addressing**, and EVERY rotary cell has
BOTH sites on:
| variant | cam_encoder | rotary sites | ladder | mean loss | vs base (paired) |
|---|---|---|---|---|---|
| ccv_base | **ON** | none | — | 0.04997 | — |
| ccv_pra | off | input+hidden | learnable | 0.04742 | −5.1%, t=−12.6, 64/64 |
| ccv_pra_fixed | off | input+hidden | fixed | 0.04633 | −7.3%, t=−12.9, 64/64; beats pra t=−11.6 |
| ccv_both | **ON** | input+hidden | learnable | **0.04562** | **−8.7%, t=−11.3, 63/64; beats pra t=−9.0, beats pra_fixed t=−5.9** |
WHAT THIS GRID DOES SUPPORT, and it is strong:
1. **Rotary addressing BEATS ReCamMaster-style feature injection, and replaces it.**
   `ccv_pra_fixed` has NO cam_encoder at all and still beats `ccv_base`, which has one,
   by −7.3%. Camera as an address outperforms camera as injected features.
2. **The two compose**: adding the cam_encoder back on top (`ccv_both`) gains a further
   −1.4 points to −8.7%.
3. Fixed ladder beats learnable in video too (third domain: NVS F25 / LLM F20/F27).
WHAT IT DOES **NOT** SUPPORT: any input-vs-hidden decomposition. There is no
hidden-only cell and no input-only cell here, so the earlier claim "THE HIDDEN ROTARY
EARNS IN VIDEO" is NOT established by these four runs, and F30b's "hidden increment"
is really the **cam_encoder increment** (`both` − `pra`). The load-bearing-memory story
(F21/F22 idle vs ccv load-bearing) still holds for the ROTARY AS A WHOLE, since
ccv forces the source view through the fast-weight update; it just cannot be attributed
to the hidden site from this grid.
TO MAKE CCV A SITE ABLATION, two runs are needed, both with cam_encoder OFF and fixed
ladders so they line up with `ccv_pra_fixed`: input-only and hidden-only. Queued as P5
(hidden-only) and P5b (input-only).
- Fixed ladder beats learnable in video too (third domain: NVS F25 / LLM F20/F27 / ccv).
- Generation eval + ReCamMaster external anchor: see F30c.
- Q10 gain variants (g03/g01) CANCELLED at step ~650 (2026-07-12 user pivot to Q11:
  frozen ReCamMaster + TTT adapter; see EXPERIMENT_QUEUE.md Q11).

### F30b: per-pair anatomy of the ccv gains (2026-07-12)
Geometry note: in MultiCamVideo all 10 cameras START at the same pose and diverge along
different trajectories — relative geometry must be measured as trajectory divergence
(mean per-frame relative rotation 0.9-48 deg across our 64 pairs), not frame-0 offset.
1. Gains are GEOMETRY-UNIFORM: spearman vs divergence ~0 for all three effects; terciles
   input +5.6/+5.9/+4.4%, full +9.4/+10.5/+7.0%, hidden increment +3.9/+4.6/+2.6%
   (small/mid/large). Mild mid-divergence peak; slight relative fade at extreme
   divergence. Lens sets (f18-f50) uniform. The rotary earns broadly, not on a
   viewpoint-outlier subset.
2. CHANNELS CO-VARY: spearman(input gain, hidden increment) = +0.86 across pairs — the
   pairs where the input rotation helps most are the same pairs where the hidden adds
   most. A common per-pair factor ("how much the pair exercises fast-weight recall")
   drives both — the video analogue of NVS F25 sub-additivity, and direct evidence for
   the shared-recall-pathway picture. Pair difficulty (base loss) correlates mildly with
   gain (+0.28).

### F30c: generation metrics @13999 (8 pairs, teacher-forcing Euler 40 steps, no CFG) + external anchor (2026-07-12)
| model | PSNR | SSIM | LPIPS | paired LPIPS vs base |
|---|---|---|---|---|
| ccv_base | 14.07±3.9 | 0.4652 | 0.6122 | — |
| ccv_pra (input, learnable) | 14.58±3.9 | 0.4847 | 0.5451 | −0.067 (t=−3.1, 8/8 pairs) |
| ccv_both (input+hidden) | 14.10±4.4 | 0.4839 | 0.5548 | −0.057 (t=−3.0, 7/8) |
| ccv_pra_fixed | 13.49±2.6 | 0.4688 | 0.5833 | −0.029 (t=−2.7) |
| **ReCamMaster step20000 (external)** | **15.71** | **0.5279** | **0.4534** | (their sampler: 50 steps, CFG 5) |
1. The rotary's perceptual gain SURVIVES generation: LPIPS is the discriminative metric
   and pra beats base on 8/8 pairs. Direction matches the 64-pair val loss (F30).
2. The hidden increment is NOT resolved at n=8 generation (both−pra LPIPS +0.010,
   t=+0.7) — val loss (t=−9.0, n=64) stays the paper-grade discriminator; generation
   metrics are the direction check. Generation also flips learnable-vs-fixed relative
   to val loss — n=8 noise, do not over-read.
3. ReCamMaster (8xH800x3days, frozen-backbone adapter recipe) clearly outranks every
   full-TTT-replacement run (LPIPS 0.453 vs best 0.545) — motivated the Q11 pivot:
   frozen pretrained weights + TTT only where ReCamMaster put its new mechanism.
   Caveats in gen_recam_anchor/metrics.json (their sampler settings, our caption).
   Eval infra: eval_ccv_generate.py + eval_recam_anchor.py; per-pair jsons + mp4s in
   lact_ar_video/outputs/eval_dev/.

## F29: hidden-normalization (hnorm/rms_rot) full verdict — a 1D-specific fix, neutral in 6D (2026-07-11)
User idea: RMS-normalize the rotated hidden dims before the hidden rotary (make the
hidden code spherical like the input q/k; F27c geometry hypothesis). Verified exact
implementations in both codebases (LLM 8a269a3, NVS c041366).
- LLM 1D (3B): rms_rot honly 18.51 on seed 42 (−0.11, briefly the champion) but 18.95
  (+0.27) on seed 137 — SEED-FRAGILE; composition with gain 0.1 (18.86), delta_only 3B
  (18.72), and rope+rms_rot stacking (19.00) all fail. gain 0.1 remains the only
  2-seed-consistent honly variant (mean −0.07).
- NVS 6D (3-seed, F25-matched): h_pra_hi+hnrot 22.668±0.151 vs anchor 22.701±0.154
  (−0.03); pra_h_hi+hnrot 22.842±0.041 vs 22.824±0.065 (+0.02). NEUTRAL in both.
- READING: the absolute-phase tax that normalization removes is a 1D pathology (no
  relative signal to pay for it); in 6D the relative signal dominates and the tax was
  never binding, so sphericalizing the code changes nothing. Q10 variant set stays
  gain-sweep only (no hnorm).

### F27e: {draw} x {budget} 2x2 complete — the sign is set by the DRAW, not the budget (2026-07-10)
ds42 0.5B pair (same 15,258-step protocol as the ds43 pair; val = ds42 head):
rope 27.92 / hpra 28.57 -> hpra−rope = **+0.65 ppl (hurt)**.
| hpra−rope | 0.5B | 3B |
|---|---|---|
| ds42 | +0.65 | +0.23 |
| ds43 | −0.26 | — |
- On ds42 the hidden rotary hurts at BOTH budgets (worse at short); on ds43 it helps.
  Together with the old-env draw (−0.19): 2 of 3 draws help, 1 hurts. The 1D hidden
  effect is a genuine coin-flip across data draws (magnitude ~0.2-0.7 ppl), while the
  input-rotary gain is stable in every measurement. Paper wording should say
  "sign varies with the data draw", not "slightly negative".
- Re-colors Q9 gen-0: the GA runs on ds42 — the UNFAVORABLE draw — and ga_honly_gain01
  still beat nope by −0.34 there. Pending its 3B confirmation, the low-gain ladder may
  be a draw-robust positive.
- Paper LLM paragraph: HOLD until ga_honly_gain01_full (3B) lands, then write the final
  story in one pass.

### F27c: input-vs-hidden asymmetry quantified + val-cache incident (2026-07-09)
- Per-position profiles of BOTH main effects are FLAT and symmetric in magnitude:
  input gain (nope-rope) = −0.009..−0.014 loss in every 512-bucket incl. bucket 0;
  hidden tax (hpra-rope) = +0.011..+0.015 incl. bucket 0. Bucket 0 predates any fast-weight
  update => in 1D both rotations act mainly through their ABSOLUTE component (the rotated
  initial readout), not distance-selective recall; the input site's absolute component is a
  useful position prior, the hidden site's is a code-scrambling tax.
- Surgery asymmetry (same seed-42 val): input rotation OFF on rope ckpt => 18.40 -> 62.8 ppl
  (+1.23 loss; catastrophic in buckets 0-1 at ~6.95, +0.10 late) and hpra ckpt => 23.20;
  hidden rotation OFF on hpra ckpt => +0.003 loss. ~400x reliance gap: the trained model is
  load-bearing on the input rotation, indifferent to the hidden one.
- Why the same absolute stamp helps at input but hurts at hidden (working theory):
  input q/k are L2-normalized dense codes on a sphere — rotation moves the point along the
  sphere, norm/score geometry intact, upstream projection + W^0 co-adapt cheaply, and the
  rotated initial readout doubles as a free absolute position encoding (this model has no
  other APE). The hidden code is silu-gated, sparse, axis-aligned, and feeds the output
  dictionary w1 directly: position-dependent pairwise mixing scrambles feature-to-column
  alignment, forcing w1 pairs toward rotation-degeneracy (capacity cost) — and its
  disambiguation value is redundant once input rotation already position-tags the addresses
  (one injective tag suffices).
- APE-vs-RoPE objection (user): in attention LLMs RoPE >> APE, so "relative buys nothing in
  1D" is too strong. Resolution: RoPE's edge in attention lives at short range (syntax,
  induction-head pointer arithmetic "previous occurrence + 1") and in length extrapolation.
  Here SWA(1024, own RoPE) owns exactly that territory; the TTT layer owns >window recall,
  which fast weights serve ASSOCIATIVELY (key->value bound at update time, no pointer
  arithmetic left for offsets to help). Scoped claim for the paper: relative position is
  cheap and vital where retrieval is positional (attention); it adds little where retrieval
  is associative and the coordinate is 1D — and it becomes the main signal when the
  coordinate is 6D geometry (NVS).
- INCIDENT (infra): the ds43 launch OVERWROTE the seed-42 val cache (filename lacked
  data_seed; the mismatch-overwrite guard in get_or_build_val_set fired on the seed-43
  stream). All F27 numbers + probe 1 predate the overwrite (valid); probe-2 first pass ran
  on the seed-43 val (levels shifted: same rope ckpt 18.40 -> 17.87, i.e., val-sample choice
  alone moves absolute ppl by ~0.5 — direct support for the datasets-stream explanation of
  the old-vs-new level shift). FIX: cache filename now carries _ds<seed> (train_small.py);
  seed-42 cache regenerated deterministically and verified (rope 2.9126, buckets bit-match).
  In-flight ds43 pair unaffected (evaluates on the seed-43 blocks, internally consistent).

## Protocol decision: the tttLRM from-scratch grid ENDS at 15,000 steps (2026-08-05)

No wave 2. `train_cam.py:49` `lr_at()` is a cosine with `total = max_opt_steps` and a
pure function of `step`, so raising `max_opt_steps` and resuming does NOT extend the
schedule: it lifts the LR from 0 back to ~5.3e-5 at step 15000, i.e. a warm restart.
The 15k endpoint and any extended endpoint would not be two points on one curve, and
neither would be reproducible from a single config. If a longer budget is ever wanted,
set `total=30000` and train from scratch. (User decision; supersedes the "wave 2 raises
max_opt_steps and resumes" note that was in `run_scratch_grid.sh`.)

Measured cost, 2 GPUs per cell, all 8 GPUs loaded (instantaneous s/step by
cumulative-average differencing over 1000 steps):

| cell | s/step | 15,000 steps |
|---|---|---|
| base | 2.705 | 11.3 h |
| in | 2.555 | 10.6 h |
| h | 2.765 | 11.5 h |
| both | 2.856 | 11.9 h |

`in` is FASTER than `base` while doing strictly more work, so GPU-placement variance is
at least 2.6 percentage points here. Read `both`'s overhead as 3-8%, the same size as
the hardware variance, not as the 5.6% the table nominally shows.

## NVS: is the rotary's lead stable THROUGH training? (seed 95, train PSNR, 2026-08-05)

Same seed and data order in all four arms; 2500-step window means. This is TRAINING
PSNR, not held out, so it bounds nothing about generalisation on its own.

| step | base | input | hidden | Both | Both-hidden | Both-input |
|---|---|---|---|---|---|---|
| 2500 | 20.187 | 20.812 | 20.891 | 21.288 | +0.398 | +0.476 |
| 7500 | 19.818 | 20.346 | 20.537 | 20.800 | +0.263 | +0.454 |
| 12500 | 20.610 | 21.012 | 21.332 | 21.433 | +0.101 | +0.421 |
| 15000 | 20.605 | 20.992 | 21.338 | 21.394 | +0.056 | +0.402 |
| 20000 | 20.779 | 21.142 | 21.515 | 21.584 | +0.068 | +0.442 |
| 25000 | 20.855 | 21.199 | 21.531 | 21.575 | +0.044 | +0.375 |
| 27500 | 21.061 | 21.338 | 21.705 | 21.704 | **-0.001** | +0.367 |

1. All three rotary arms beat base in EVERY window, start to finish. No rank inversion.
2. Both > input holds throughout at a stable +0.35 to +0.51, no trend.
3. **Both > hidden decays monotonically to zero.** +0.398 -> +0.056 -> -0.001.

Held-out confirmation of (3), Both minus hidden-only paired over the same 256 scenes
(view sweep checkpoints, seed 95):

| views | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| delta | **-0.225** | +0.073 | +0.254 | +0.255 | +0.308 |
| t | -10.44 | +3.70 | +16.64 | +16.29 | +21.91 |
| Both wins | 56/256 | 155/256 | 219/256 | 220/256 | 233/256 |

At 4 views hidden-only BEATS Both outright. At 8 views -- the standard protocol -- the
+0.073 gap is a fifth of the 0.35 dB seed-noise floor (F18), i.e. a tie despite
t=3.70. Both only separates from 16 views on. Consistent with Q4's sub-additivity
(0.60 + 0.96 = 1.56 but full = 1.08): the two address spaces overlap, and the second
site pays only once the address space is actually the bottleneck. Same mechanism as
F40's "NoPE saturates at 16 views".

CONSEQUENCE: the 30k budget is a stopping point, not convergence (train.py's own
default is 80000, and base_s95 is still climbing in its last window: 20.855 -> 21.061).
Both's lead over hidden decaying to zero AT that stopping point is why GROUP E's 80k
run includes a hidden-only arm.

## Q31 quarantine: `q31_attnnope_nope_s43` diverged mid-training (2026-08-05)

Do not average this cell in. It tracks its siblings exactly for the first half, then
turns around and climbs:

| step | 2k | 14k | 26k | 38k | 50k | 62k | 74k | 86k |
|---|---|---|---|---|---|---|---|---|
| nope_s43 | 116.76 | 32.51 | 27.25 | 24.67 | 22.78 | 26.20 | 28.50 | **30.40** |
| nope_s42 | 118.84 | 32.80 | 27.49 | 24.90 | 22.94 | 21.37 | 20.26 | 19.72 |
| in_s43 | 116.35 | 32.34 | 27.10 | 24.58 | 22.62 | 21.06 | 19.95 | 19.42 |

A mid-training instability at 50-62k, not a bad init and not a data-order effect:
through 50k it is indistinguishable from the cells that finished. Every other Q31 cell
lands in 19.38-19.72, so folding 30.40 into a mean would manufacture a large fake
"NoPE is much worse under attn_nope" effect out of one blown-up run.

Repair: `q31_attnnope_nope_s43b`, data_seed 43 (so it stays paired with in/h/both at
that data seed), init seed 42 -> 44. The diverged run is KEPT as evidence.

**If the rerun also diverges, that is the result.** A model with no explicit positional
code anywhere (attn_nope + ttt_nope) being less trainable is a real property and should
be reported as instability, not as a perplexity. Do not keep drawing seeds until one
finishes clean.

Q31 completion status otherwise: 8/8 cells at 3B tokens, step 91552, two data seeds.
Q29 is 7/7 complete with `q29_paired.json` computed. Only Q30 (13 of 15 cells for
DIMS='2 4 6') remained from node3's queue.

## F45: tttLRM from-scratch grid, 15,000 steps — every rotary arm beats NoPE, but the
## two sites do NOT compose here (2026-08-06)
DL3DV-140, 8 input views, 272x480, d768/24L, from scratch, cosine to 0 at 15,000 (the
annealed endpoint; there is deliberately no wave 2). Rotary arms at NVS ladder width
(num_freqs 64 / num_freqs_h 256), which removes the 25%/8.2% coverage confound of the
fine-tune round. Paired per scene, n=140.

| arm | PSNR | SSIM | LPIPS | dPSNR vs base | t | improved |
|---|---|---|---|---|---|---|
| base (NoPE) | 15.240 | 0.3682 | 0.6508 | — | — | — |
| **in** (input site) | **15.691** | **0.3825** | 0.6158 | **+0.451** | +16.90 | 135/140 |
| h (hidden site) | 15.672 | 0.3820 | **0.6132** | +0.433 | +15.56 | 134/140 |
| both | 15.532 | 0.3799 | 0.6200 | +0.293 | +11.07 | 124/140 |

1. **The rotary works from scratch on 3D reconstruction.** All three arms beat NoPE at
   t=11-17 on 124-135 of 140 scenes, and LPIPS improves on 139-140/140. This is the
   result the fine-tune round (F39) structurally could not produce: the released
   checkpoint was already converged, leaving ~0.08 dB of headroom, so that design could
   only ever measure what the rotary COSTS.
2. **in and h are statistically indistinguishable** (+0.451 vs +0.433) — unlike NVS,
   where the hidden site carries most of the fixed-ladder gain.
3. **both is WORSE than either single site** (+0.293), not merely sub-additive. Adding
   the second site here REMOVES about a third of the gain.

Point 3 is the same direction as the NVS through-training result (Both's lead over
hidden-only decays to zero by 27.5k) and the same direction as Q4's sub-additivity, but
stronger: in NVS the sites still add a little at 8 views, here they subtract. Both
observations say the two address spaces overlap; tttLRM at full ladder width is where
the overlap turns costly.

CAVEAT ON ABSOLUTE LEVEL: 15.2-15.7 PSNR is NOT comparable to the 25.062 anchor or to
the fine-tune grid. Those use 32 input views at 536x960; these are 8 views at 272x480
from scratch at 15k steps. The four numbers are comparable only to each other, which is
all the four-arm contrast needs.

## F46: PRoPE-style camera/image budget split on NVS (camimg, seed 95, 2026-08-06)
Standard protocol. Budget-matched to qk_rope_cam by construction and by measurement:
6*F_cam + 2*F_img = 6*10 + 2*33 = 126 pairs = 6*21, 252 of 256 head dims rotated in
both. So this is purely a question of ALLOCATION.

| variant | PSNR | LPIPS |
|---|---|---|
| base | 21.825 | 0.2874 |
| pra_hi (input site, 100% camera) | 22.333 | 0.2751 |
| **camimg (input site, 50% camera + 50% image)** | **22.451** | **0.2738** |
| pra_h_hi (input + hidden, 100% camera) | 22.797 | 0.2685 |

Paired per scene, n=256:
- camimg - pra_hi: **+0.117 PSNR (t=+8.51, 181/256), -0.0013 LPIPS (t=-3.96)**
- camimg - base: +0.625 (t=+19.23, 230/256)
- camimg - pra_h_hi: **-0.346 (t=-21.19, 17/256)**

1. **Spending half the input-site budget on in-view 2D position beats spending all of
   it on camera** — small (+0.117) but consistent, 181 of 256 scenes, t=8.51. It is
   below the 0.35 dB seed-noise floor (F18) as an absolute PSNR gap, so a 3-seed
   confirmation is needed before this goes anywhere near the paper; the paired t is
   about the same checkpoint pair, not about seed robustness.
2. **It does not approach the second SITE.** Adding the hidden site is worth +0.464
   over pra_hi; reallocating half the input budget to image coordinates is worth
   +0.117, i.e. a quarter as much. The site axis dominates the allocation axis.
3. Partially replicates F34's finding that PRoPE's gain in this stack came from its
   image-coordinate ropes, now WITHOUT the projective transform that cost -0.294.

## Infrastructure bug: eval steps ran without the compile caches (2026-08-06)
`launch_exp.sh` exports TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR to lustre, but
run_camimg.sh, run_budget80k.sh and run_groupA.sh invoked `eval.py` as a separate
process that did not inherit them. Default /tmp/torchinductor_* is a noexec tmpfs, so
inductor died with "failed to map segment from shared object" AFTER training completed.
camimg_s95 hit this and lost its eval; the four 80k runs and seven Group A runs would
each have hit it hours later. Fixed by exporting in all three scripts. Any new launcher
that calls eval.py directly needs the same three lines.

## F47: NVS budget sensitivity — 30k is not convergence, and the gaps SHRINK but hold
## (seed 95, 2026-08-06)
Same protocol as the 30k table but 80,000 steps, warmup scaled 1500 -> 4000 so the
cosine keeps its shape, `lpips_start` left absolute at 5000. Comparable only to itself.

| arm | 30k PSNR | 80k PSNR | 30k d vs base | 80k d vs base (t, improved) |
|---|---|---|---|---|
| base | 21.825 | 23.260 | — | — |
| input | 22.333 | 23.619 | +0.508 | +0.359 (t=+18.3, 234/256) |
| hidden | 22.724 | 24.027 | +0.899 | +0.767 (t=+30.6, 249/256) |
| both | 22.797 | **24.071** | +0.971 | **+0.811** (t=+29.5, 247/256) |

LPIPS at 80k: base 0.2176, input 0.2056, hidden **0.1984**, both 0.1995.

1. **30k was nowhere near convergence.** Every arm gains 1.3-1.4 dB from the extra
   50k steps. Any statement of the form "the model has converged" at 30k is wrong.
2. **The rotary's advantage survives, reduced.** Both keeps +0.81 dB at t=29.5 on
   247/256 scenes. NoPE does NOT catch up. But the gap shrank ~17% (+0.971 -> +0.811),
   so part of what the rotary bought at 30k was faster convergence, not only a better
   endpoint. Report the 30k gap as budget-dependent, never as budget-invariant.
3. **The hidden-overtakes-Both question is answered: no.** Both - hidden goes
   +0.073 (t=3.69) at 30k to +0.044 (t=2.44) at 80k. The lead narrows further but does
   not cross. So the two sites still do not COMPOSE meaningfully in NVS at 8 views --
   the second site is worth about 0.04-0.07 dB, a fifth of the seed-noise floor -- but
   the extrapolation that hidden-only would win was wrong.
4. Note hidden takes the best LPIPS at 80k while both takes the best PSNR.

## F48: tttLRM 3D-reconstruction view sweep, 4-32 input views (step 15000, n=140)
Evaluation only: one checkpoint per arm read at five input scales. Trained at 8 views.
View selection is one rule across all five points (kmeans, random_state=0); the test
list precomputes 4/16/32 and v8/v24 recompute the identical quantity.

| arm | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| base | 14.152 | 15.240 | 15.446 | 15.295 | 15.117 |
| in | 14.452 | 15.691 | 15.880 | 15.666 | 15.424 |
| h | 14.408 | 15.672 | **15.975** | **15.823** | **15.617** |
| both | 14.006 | 15.532 | 15.526 | 15.210 | 14.955 |

Paired delta vs base (t):

| arm | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| in | +0.300 (11.6) | +0.451 (16.9) | +0.435 (13.0) | +0.370 (9.7) | +0.306 (7.9) |
| h | +0.257 (8.7) | +0.433 (15.6) | +0.529 (16.8) | +0.528 (14.0) | **+0.500 (13.1)** |
| both | **-0.145 (-4.8)** | +0.293 (11.1) | +0.080 (2.1) | -0.086 (-1.9) | **-0.162 (-3.6)** |

1. **The single sites separate along the view axis exactly as the two-axis thesis
   predicts.** `in` peaks at 8-16 views and decays (+0.451 -> +0.306); `h` keeps
   climbing to 16 and HOLDS out to 32 (+0.529, +0.528, +0.500). By 32 views the hidden
   site is worth 1.6x the input site. This is the cleanest within-task evidence we have
   that the hidden site is what scales with input load.
2. **`both` is actively harmful at the ends** — negative at 4 views (t=-4.8) and at 24
   and 32 (t=-1.9, -3.6). It only helps in a narrow band around the trained point.
3. Every arm including base peaks at 16 views and declines after, unlike NVS where NoPE
   plateaued. Different backbone, different saturation shape; do not merge the panels'
   narratives.

Figure regenerated at paper_overleaf/fig1_input_scale.pdf with both panels complete.

## F50: Q32 LaCT-LVSM on DL3DV — changing ONLY the data flips `both` from the best
## arm to a harmful one (seed 95, 2026-08-06)
Same LaCT-LVSM, same configs, same 30k/bs16/lr1e-4/8+8/256x256/LPIPS-from-5k protocol
as the RE10K table; DL3DV instead of RE10K. This is the lever F45 could not isolate:
tttLRM differed in backbone (24L vs 6), head (Gaussians+depth vs RGB), ladder AND data
at once. Here only the data moves. Eval on the SAME 140-scene DL3DV test split F45 used,
8 uniform inputs / 4 midpoint targets. Paired per scene, n=140.

| arm | PSNR | LPIPS | dPSNR vs base | t | improved | dLPIPS (t) |
|---|---|---|---|---|---|---|
| base (NoPE) | 16.398 | 0.4772 | — | — | — | — |
| input | 16.408 | 0.4776 | **+0.010** | 0.55 | 68/140 | +0.0004 (0.6) |
| **hidden** | **16.537** | **0.4746** | **+0.139** | 7.81 | 104/140 | **-0.0026 (-4.3)** |
| both | 16.389 | 0.4837 | **-0.009** | -0.41 | 61/140 | +0.0065 (9.2) |

**THE NUMBER: both - max(input, hidden) = -0.148 dB, t = -9.74, both wins 27/140.**

1. **`both` FAILS TO COMPOSE ON DL3DV, with the tttLRM-side confounds removed.** On
   RE10K at this protocol `both` is the BEST arm (+0.971 vs base, and +0.073 over
   hidden, F47). Here it is -0.009 vs base and -0.148 BELOW hidden. Same backbone, same
   head, same ladder, same optimizer, same budget — only the dataset changed. The
   pre-registered reading is therefore the first one: **data geometry is the lever**,
   and "when do the two sites compose" is a claim about camera baseline, not about
   tttLRM's architecture.
2. **The INPUT site dies at wide baseline.** +0.508 on RE10K -> **+0.010 (t=0.55)** here,
   a clean null. The hidden site survives at +0.139 (t=7.81) and takes the only LPIPS
   improvement. hidden - input = +0.129 (t=9.28, 113/140). Contrast F45, where on
   tttLRM/DL3DV the two single sites were statistically indistinguishable (+0.451 vs
   +0.433) — so backbone still shapes WHICH single site wins, even though it is not what
   decides composition.
3. Consistent with the wrap account: at RE10K's ~7 deg between-view angle ~38% of the
   ladder wraps between views vs ~50% at DL3DV's, and a second site doubles that dose
   without adding information. Directional support, not proof — this grid has one
   geometry contrast, not a dose-response curve. Q33 (gObjaverse, ~91 deg) is running to
   add the third point.

PROTOCOL NOTES, all shared by the four arms, none comparable outside this grid:
- 10,125 train scenes vs RE10K's 66k -> ~47 epochs over 30k steps vs ~7.
- Images stored pre-sized to cover 256 (one extra resample vs RE10K's native 640x360).
- Absolute PSNR (16.4) is NOT comparable to the RE10K table (21.8) or to F45's tttLRM
  numbers (15.2); the four numbers here are comparable only to each other.
- Eval-window geometry measured through the actual loader, as the angle between camera
  forward axes over the 12 views served: median 34.5 deg, mean 49.0 (RE10K ~7 at the
  same protocol). A different convention in the launch brief reports 20.6 deg for the
  same data; both say the geometry contrast survives the 128-frame window, which is the
  load-bearing part.
- Throughput note: DL3DV and RE10K train at the SAME speed once matched by phase
  (~11.3 it/s before LPIPS, ~4.9 after, both datasets). An earlier read of ~2.3x was an
  artifact of comparing DL3DV's pre-LPIPS rate to RE10K's post-LPIPS rate.

## F49: Q31 NoPE repair diverged again, worse — the instability IS the result
`q31_attnnope_nope_s43b` (data_seed 43, init seed 44) ended at **ppl 526.0** against
the first attempt's 30.4 and its siblings' 19.4-19.7. Per the rule written down before
this run: a second divergence is reported as instability, not as a perplexity, and no
further seeds are drawn. Two of two NoPE cells at data_seed 43 blew up while all six
rotary cells at both data seeds finished clean.

Reading: with `attn_nope` the model has no explicit positional code anywhere, and that
configuration appears to be genuinely less trainable at 3B tokens. That is a property
worth stating, and it makes the ds42 nope number (19.69) the one usable NoPE point.
F44's conclusions rest on ds42 and are unaffected; what is NOT available is a two-seed
confirmation of the NoPE arm.

## Q30 interim (d=2/4/6 complete at 800M tokens; d=3/5 at ~70%): one cell learns,
## and it is exactly the predicted one (2026-08-06)
N-dimensional tensor recall, single seed 42, answer_acc over held-out blocks; content
vocab 1000 so CHANCE = 0.1%.

| d | base | in flat | in nd | h flat | **h nd** |
|---|---|---|---|---|---|
| 2 | 0.05% | 0.04% | 0.06% | 0.07% | **0.90%** |
| 4 | 0.07% | 0.07% | 0.07% | 0.07% | 0.07% |
| 6 | 0.09% | 0.09% | 0.13% | 0.16% | 0.11% |

1. **The ONLY configuration above chance anywhere in the grid is hidden + true 2-D
   address**: 9x chance and still climbing at budget end (0.65 -> 0.90% over the last
   4k steps; val_loss 6.843 vs 6.908 for all four others, which are mutually
   indistinguishable). The input site given the SAME nd address stays at chance. As a
   sign this is the dimensionality thesis's prediction exactly: the hidden site is
   what exploits a multi-dimensional address.
2. **d >= 4 is a floor effect, not a refutation.** Every arm sits at chance, so those
   cells cannot distinguish anything at this budget. The pre-registered "peak" cannot
   be located from a floor.
3. Honest limits: single seed; the one live cell is at 0.9% ABSOLUTE accuracy; and the
   task at 800M tokens is simply too hard for every other configuration. The clean
   claim available today is the d=2 ordering (h_nd >> everything), not a curve over d.

NEXT (cheap, decisive): extend q30_h_d2_nd + q30_in_d2_nd + q30_h_d2_flat to 3B
tokens (the curve is still steep), and consider an easier variant (smaller grid or
content vocab) so d>=3 lifts off the floor before comparing sites there.

## F52: ladder width is a MODULATOR, not the cause -- the coverage comparison lands
## (tttLRM from scratch @ step 10000, only the ladder differs; 2026-08-06)
The nf16 grid's checkpoints survived under outputs/scratch_*_nf16 (its results were
never recorded in this dossier -- repaired here). Same protocol, same seed, same step;
nf16 = F16/F21 (coverage 25%/8.2%, oversampling 1.03x/1.35x, i.e. NVS-like density),
wide = F64/F256 (100%/100%, 4.1x/16.5x).

| | base | in | h | both | both - best single |
|---|---|---|---|---|---|
| nf16 | 15.007 | +0.505 | +0.529 | +0.432 | **-0.097** (t=-5.45, 42/140) |
| wide | 15.104 | +0.556 | +0.496 | +0.309 | **-0.247** (t=-13.14, 17/140) |

1. **`both` fails to compose at BOTH widths.** The near-critical nf16 ladder does not
   rescue it, so oversampling/coverage is NOT the cause of the failure. The density
   hypothesis in DENSITY_PROBE.md is refuted as the primary account, and the
   scratch_d_* probe cells are now unnecessary.
2. Width MODULATES the size of the failure: -0.097 -> -0.247 as the ladder saturates.
   Consistent with the wrap account (a wider ladder doubles a larger dose) but only as
   a second-order effect.
3. Together with F50 the picture is: **data geometry decides WHETHER the sites
   compose** (RE10K +0.07 vs DL3DV -0.148 at identical width on the same backbone);
   **ladder width scales HOW BADLY** they interfere once they do not (-0.097 vs
   -0.247); and the unrotated-dimension hypothesis is dead (nf16 leaves 75%/91.8%
   unrotated and still fails).

## F53 (eval-only half): view DENSITY alone restores composition on LVSM/DL3DV --
## and the crossing sits where the nearest-neighbour angle passes RE10K's (2026-08-06)
The F50 checkpoints (trained at 8 views) re-evaluated at 4-32 input views, same 140
scenes, evaluation only. both - best single site, paired per scene:

| v | 4 | 8 (=F50) | 16 | 24 | 32 |
|---|---|---|---|---|---|
| both - best | -0.147 | -0.148 | -0.010 | +0.013 | **+0.092** |
| t | -7.57 | -9.74 | -0.62 | +0.79 | **+5.55** |
| NN angle | ~18 deg | 12.8 | 8.0 | ~5.8 | 4.7 |

(median pairwise angle stays ~39-45 deg throughout; only the nearest-neighbour
angle moves. v=8 row is F50 itself.)

1. **The composition failure REVERSES with evaluation-time view density alone** --
   no retraining. The user's hypothesis (more views -> overlapping poses -> camera
   addressing matters more) is supported on LVSM, and the phases learned at 8 views
   already suffice; what was missing at 8 views was the geometry, not the training.
2. **The sign flips where the nearest-neighbour angle crosses ~7-8 deg** -- i.e.
   RE10K's regime (7.2 deg). Deficit is flat (~-0.15) while NN >= 13 deg, zero at
   NN = 8.0, positive at NN <= 5.8. So the geometry statistic that governs
   composition appears to be the NEAREST-NEIGHBOUR baseline, not the median pair:
   composition needs some near-duplicate views to anchor, and the wide typical pair
   does not preclude it.
3. **Backbone flips the direction of the view-count response**: the same eval-only
   manipulation on tttLRM (F48) made `both` monotonically WORSE toward 32 views.
   LVSM recovers, tttLRM does not. Which single site wins AND how view density acts
   are backbone properties; whether composition is possible at all tracks geometry.
4. Single seed today. Seeds 137/211 for all four arms (plus their v32 evals) are
   training now; node2's Q34 (TRAINED at 32 views) completes the 2x2 and tests
   whether training at density adds anything on top of evaluating at density.

Consequence for the pending paper revision: state the geometry claim in terms of the
nearest-neighbour baseline, not the median pairwise angle. RE10K-vs-DL3DV at 8 views
differs in BOTH statistics, so F50 alone could not separate them; this sweep can,
because v moves only the NN statistic and the sign follows it.

## Context check: the tttLRM PAPER's curriculum never trains the scene model below 16
## views (arXiv 2602.20160, App. B.1; checked 2026-08-06)
Scene-level (the DL3DV model all our tttLRM work derives from): three resolution
stages, 144x256 -> 288x512 -> 540x960, and "we train the model across 16 to 64 input
views"; the final stage trains "with 32 input views ... then 16 to 64 input views"
(verbatim; one summary pass read the stage-3 step counts as 55K+11K and another as
5K+1K -- re-verify against the PDF before citing step counts anywhere). Object-level
GS model: "8 views as input and another 8 views as supervision" at every stage.

Consequence for reading F45/F48: our from-scratch grid trains the SCENE task at 8+8
views -- the OBJECT recipe's view count, below the scene curriculum's minimum of 16.
So the composition failure (F45) and the worsening-with-views eval response (F48) were
measured in a view regime the paper's scene model never occupies. Combined with F53
(composition on LVSM/DL3DV flips positive once the nearest-neighbour angle drops to
RE10K's), the 8-view choice is a real threat to external validity: the published
tttLRM lives in the dense regime where composition should be easiest. A from-scratch
cell at 16-64 mixed views (the paper's actual recipe) is the missing measurement;
decision deferred until Q34 (LVSM trained-at-32) reports.
