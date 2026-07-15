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
| variant | mean loss | vs base (paired) |
|---|---|---|
| ccv_base | 0.04997 | — |
| ccv_pra (input, learnable) | 0.04742 | −5.1%, t=−12.6, win 64/64 |
| ccv_pra_fixed (input, fixed ladder) | 0.04633 | −7.3%, t=−12.9, 64/64; beats pra t=−11.6 |
| ccv_both (input+hidden) | **0.04562** | **−8.7%, t=−11.3, 63/64; beats pra t=−9.0 (54/64), beats pra_fixed t=−5.9** |
- THE HIDDEN ROTARY EARNS IN VIDEO once the task forces cross-camera information through
  the fast weights (ccv: source view enters ONLY via fast-weight update; target camera
  differs). F21/F22 neutrality was the idle-memory regime, exactly as CAMCTRL_DESIGN
  hypothesized. Paper video story upgrades from "neutral boundary" to "neutral when the
  memory carries no exclusive workload, large when it does".
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
