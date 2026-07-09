# Writing Brief — Abstract + Introduction + Method rewrite

Source of truth for all writing agents. Do not invent results; use only what is here.

## The problem (from instruction.md)
- TTT (test-time training) layers replace attention with linear-time fast-weight memories.
- Attention has mature camera conditioning (CaPE, GTA, PRoPE, RayRoPE); TTT has none.
- PRoPE-style relative encodings rely on attention's bilinear logits; TTT layers are nonlinear
  (SwiGLU fast weights updated by gradient descent), so the transfer is not obvious.
- Goal: camera (and generally positional) conditioning for LaCT-style TTT layers, applicable at
  every TTT layer, minimal compute overhead, ideally relative / SE(3)-friendly.

## The key theory (must be the centerpiece of Method)
Lemma (inner-product addressing): fast-weight updates are sums of outer products; a query meeting
an outer product contracts through an inner product ( x(aᵀb) = ⟨x,a⟩b ). Hence EVERY interaction
between a query and update-written content passes through inner products — the one algebraic
hook RoPE-style relative encodings need. Readout decomposition (one-chunk regime):
  o_j = h⁰(q_j)W1⁰  [init readout]
      + Σ lr_i ⟨h(q̃_j), h(k̃_i)⟩ v_i  [value retrieval — DOMINANT channel, lives in HIDDEN space]
      + Σ ⟨q̃_j, k̃_i⟩ c_ij  [gate corrections — lives in INPUT space]
      + O(ΔW²)
Two independently rotatable channels → two rotary sites:
1. input rotary (q/k, post-L2-norm) — relativizes the gate-correction channel; equals what prior
   work (incl. LaCT authors' own fw-RoPE in LLM/video) already does.
2. HIDDEN rotary (h-PRA): rotate the SwiGLU hidden activation between h and W1, on write and read;
   backprop uses the inverse rotation. Relativizes the dominant value-retrieval channel. This
   channel DOES NOT EXIST in attention (no hidden layer between logits and values) — the core novelty.
3. omega_map: learnable linear phase maps θ = (Ω0 + ΔΩ)π (zero/tilt init). Exactly preserves
   relativity (θ_i − θ_j = Ω(π_i − π_j)); lets phase atoms leave the coordinate axes. Only helps
   for multi-dimensional coordinates (6D camera rays, 3D video grids); degenerate for 1D text.

## Final recipe
q/k input rotary (Plücker 6-coord, F=21) + hidden rotary (F_h=42, ~half of hidden dims) +
learnable phase maps with tilt-0.1 init. Overhead <0.1% FLOPs, +0.01% params. Orthogonality is
load-bearing (norm-preserving; compatible with L2-norm, Muon, weight-norm).

## Results to cite (final numbers)
- NVS (RE10K, LaCT-LVSM 6L/d256/p16, 30k iters, 256-scene eval, 3 seeds vs 3 seeds):
  baseline 21.745 ± 0.196.
  - **Headline (F24b): depth-3 fast weights + one rotary per address space (fw3l_rot3)
    23.439 ± 0.022 (+1.694 dB)**, LPIPS 0.2478. Depth-3 alone is baseline-level (21.868, F24):
    depth only pays when every address space is relativized — "one rotary per address space".
  - 2-layer full recipe (learnable phase maps, omega_r): 22.971 ± 0.088 (+1.226 dB),
    LPIPS 0.2929 → 0.2613.
  - Main fixed-ladder ablation (F25, 3 seeds, F21/F_h42, no learnable freqs):
    full (input+hidden) 22.824 ± 0.065; hidden-only 22.701 ± 0.154; input-only 22.333;
    hidden channel carries most of the gain; channels sub-additive at saturated ladders
    (vs small-ladder additivity F9: +0.41 / +0.46 / +0.77).
  - Relative-vs-absolute isolation (F23): per-scene random phase stamps ≈ baseline (−0.11,
    within seed noise); full recipe − random stamps = +1.34 dB → the gain is carried by the
    relative component; absolute residue is benign (supports the W⁰ functional-prior claim, F12).
  - 40+ total runs; failed axes documented (projective transplant, value transport, feature
    injection, optimizer conditioning, per-view chunking) — "the attention recipe does not
    transplant; the channel structure does."
- LLM (200M, 3B tokens fineweb-edu, identical data order): original (with fw-RoPE) ppl 19.32 →
  h-PRA 19.13 (−1.0%); NoPE 19.56. Gain equals the entire NoPE→RoPE gap, additive on top.
  omega_map 1D: neutral (19.35) — confirms multi-dim boundary condition.
- Video (Wan1.3B AR attn-only finetune, MultiCamVideo, deterministic noise, paired per-step):
  h-PRA exactly neutral at 4100 steps (Δ +0.000000, t=0.0) AND at 20k steps (F22, second half
  t=−0.1) — boundary condition confirmed, not budget-limited. Honest mechanism framing (F21
  correction): 6 update chunks/seq, but the memory-exclusive workload share is thin (SWA covers
  adjacent-frame redundancy); gains require the fast-weight memory to be load-bearing.
  2-of-3 tasks improve; no task is hurt. (ccv grid — camera-controlled generation where memory
  IS load-bearing — relaunched 2026-07-09, results pending.)
- Negative-result forensics available in RESULTS_DOSSIER.md (F1–F25).

## Framing directions the user values
- TTT/SwiGLU-specific structure exploited (channels; no attention analogue for h-PRA)
- deep math intuition PRoPE-style; relative; SE(3)-friendly; minimal architecture change
- differentiation from PRoPE/GTA/RoPE; generality beyond cameras (LLM, video)
- name: PRA (Plücker Rotary Addressing) for NVS instance; the general pattern = rotary fast-weight
  addressing (input + hidden channels + learnable phase maps).

## Readability requirements (user's revision instructions)
- Current Method (overleaf_paper/prelim_method.tex) is too hard to read. Rewrite for clarity.
- Techniques: same meaning restated in different sentences; consistent terminology across
  sentences; deliberate repetition of the important points; concrete running examples; forecast
  sentences before math; interpretation sentences after every equation.
- Process: 10 diverse versions (different expository strategies) → pick 3 most understandable →
  iterative revision until readability converges.

## Terminology rule (2026-07-06, user instruction)
Follow LaCT paper terminology wherever it exists: "update" (not write) for the
fast-weight gradient step, "apply" (not read) for using fast weights on queries.
Mirror other LaCT terms (fast weights, chunk, per-token lr) as in their paper.
