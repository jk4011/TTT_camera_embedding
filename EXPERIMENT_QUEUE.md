# Experiment Queue

Pending experiments, in priority order. NVS small config = L6/d256/p16, 30k
iters, bs16, ~1.6h per run on one B200.

**Environment (2026-07-09)**: 8-GPU B200 batch node; Claude (remote session
"ttt-batch") is the sole executor — it launches runs directly as its own
background tasks (completion notifications) and coordinates GPUs via
`lact_nvs/outputs/.gpu_locks/`. The `26msit001_T_B` workspace was lost with all
its contents (old conda env, video working clone, ccv checkpoints+logs,
MultiCamVideo original); everything has been rebuilt under `26msit001_A`
(venv envs/lvsm + repo .venv_llm, datasets/ for Wan ckpt + MultiCamVideo).

## Q1. Absolute-adaptation probe  [DONE 2026-07-07 -> F23: 21.634 vs base 21.745 (within noise); PRA-relative isolates +1.34 dB]
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

## Q2. Deeper fast weights: one rotary per address space at depth 3  [DONE 2026-07-07 -> F24/F24b: fw3l_rot3 23.439+-0.022 (3 seeds), new headline; depth alone worthless, third rotary site earns +0.13]
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

## Done / superseded
- Q3 (ViT3 Stage-1, TTT-RoPE on plain ViT3-T): DROPPED per user decision 2026-07-07 (ImageNet download not worth it; NVS/video/LLM evidence sufficient for the paper).
- v20k long-budget video ablation -> F22 (h-PRA exactly neutral at 20k).
- ccv 3-run grid (base/pra/both), launched 2026-07-07 on the T_B clone: **LOST mid-run in the
  T_B workspace loss** (checkpoints and logs unrecoverable) -> superseded by Q5 below.

## Q5. ccv grid relaunch: base/pra/both/pra_fixed together  [PREPPING 2026-07-09; GPUs 2-5]
Replaces both the lost 3-run grid and the original pra_fixed add-on. All FOUR variants
(abl_ccv_base / abl_ccv_pra / abl_ccv_both / abl_ccv_pra_fixed) relaunch together via
`lact_ar_video/run_ccv_grid.sh` on GPUs 2-5: paired per-step comparison only needs shared
data order + deterministic noise WITHIN a grid, so the new pair index (new MultiCamVideo
extraction, same index_seed=42) is fine. 20k steps, ~46h each.
Rebuilt dependencies (all under 26msit001_A now): .venv_llm = torch 2.9.1+cu130 +
flash-attn 2.8.3 (source-built against system nvcc 13.1 — cu128 torch cannot build it);
Wan ckpt + MultiCamVideo at jinhyeok/datasets/ (HF re-download, 333 GB).
Sequence: dataset __main__ sanity -> 60-step sanity x4 (IMPL_SPEC_CCV.md checks) -> 20k launch.

## Q4. Rigorous 3-seed ablation at best fixed-ladder setting  [DONE 2026-07-09 -> F25: full 22.824+-0.065 / w-o input 22.701+-0.154 / w-o hidden 22.333 (s95). pra_hi s137+s211 remain — first items in BATCH_QUEUE.txt, auto-run by the batch daemon. NOTE: all pre-reset checkpoints were lost, so this was a full 3x3 retrain in the rebuilt env; fw3l_rot3 s95 re-run reproduced 23.439 exactly (env validated).]
Requested 2026-07-07. No learnable frequencies; matched ladders (input F=21, hidden F_h=42).
- Runs (seeds 95/137/211): full = pra_h_hi (qk_rope+h_pra); w/o input = h_pra_hi (new config,
  F_h42 only); w/o hidden = pra_hi (F21 only). Baseline (no PE) 3 seeds already exist.
- Reuse: pra_h_hi s95 (22.836/0.2690), pra_hi s95 (22.389/0.2753) => 7 new runs (~7h).
- Deliverable: mean +- std table for the paper's main ablation.

## Q6. Gateless 2-layer-MLP fast weights (inner-model generality)  [DONE 2026-07-10, 3 seeds -> F26 FINAL: base 20.500+-0.041 / rot2 22.477+-0.099 = +1.977 gap (> SwiGLU +1.08); rotated plain MLP beats input-only SwiGLU at 3-seed rigor; gate worth +1.25 base capacity]
Requested 2026-07-09 (user): show the recipe generalizes across inner models —
the paper's Method is written on a gateless 2-layer MLP, so back it directly.
- f(x) = silu(x W0) W1 (no gate branch); inter_multi 3 for exact fast-weight
  param parity with SwiGLU x2 (393,216). Sites: input (F=21) + single hidden
  (d_h=768 -> F_h=64, budget rule). cam_mlp2_base vs cam_mlp2_rot2, param-neutral
  up to rotary gains (+510/layer).
- Kernel verified vs autograd (bf16 tol, muon-0 transform replicated).
- Prediction: rot2 - base gap ~ the SwiGLU fixed-ladder gap (~+1.1), showing the
  channel structure (not gating) carries the effect. Seeds 137/211 after s95.

## Q7. LLM 2x2 input/hidden rotary ablation  [DONE 2026-07-09 -> F27: rope 18.40 / nope 18.62 / hpra 18.64 / honly 18.85. Input fw-RoPE −1.2% ppl replicates; 1D hidden rotary HURTS in rebuilt env (sign flip vs F19, same code — F27b audit: no bug; flat positional tax on the initial readout, near-zero relative gain in 1D; datasets-lib stream reorder = only surviving old-vs-new explanation).]

## Q9-EXT  [CLOSED 2026-07-11 per user decision. Final: rope NOT beaten hidden-only.
Robust ceiling of the 1D hidden channel = gain-0.1 gentle ladder, −0.07 ppl (2-seed) vs
rope's −0.22. ~15 3B runs + ~25 shorter runs; every deviation (gain 0.03/0.2, theta100,
frac 25/75/100, delta, hnorm-rms/rms_rot, compositions, stacking x2) failed at 3B or was
seed-fragile. Deliverables: F28 (anatomy of what wins), F29 (hnorm = 1D-specific fix,
neutral in 6D), the 45k-anti-predictivity methodology finding. See lact_llm/ga_honly/LEDGER.md.]

## Q9-EXT-ARCHIVED. STANDING GOAL (user, 2026-07-10): iterate until honly BEATS input rope at 3B
Target: honly variant <= 18.40 ppl (rope) on the full 3B ds42 protocol. Current best:
gain0.1 = 18.53 (nope 18.62, plain honly 18.85). Gap to close: −0.13.
Fitness protocol v4: 45k-step (1.5B, ~3h) runs, fitness = mean matched-step gap vs the
nope reference over the last 10k steps (gap curves are smooth to ~±0.01 — decision-grade
for 0.1-scale effects, unlike 20k endpoints). 3B full runs only for final confirmation.
Gen-3 candidates (launch as GPUs free): (a) compressed ladder theta100 x gain0.1 — all
48 pairs active in the 63..6.3k-token band (at theta 1e6 most pairs are frozen within
the 4096 window); (b) gain 0.2 and/or 0.3 (3B curve suggests optimum may sit above 0.1);
(c) frac 0.75 x gain 0.1. [learnable hidden ladder EXCLUDED per user 2026-07-10 — do
not run learnable-frequency variants; consistent with F20 (1D degeneration) and the old
'full' result.] Honest note: rope's −0.22 comes from near-tax-free input-site
collision avoidance; beating it hidden-only is a stretch goal — win or lose, the search
maps the 1D ceiling of the hidden channel (paper value either way).

## Q11. ReCamMaster-frozen + TTT-adapter 2x2  [DONE 2026-07-12 -> F31: input rotary -0.72% t=-6.15 on the frozen backbone; hidden adds nothing over input (flip vs F30); absolute quality needs longer training]
User (2026-07-12): the full-TTT-replacement ccv architecture caps quality too low —
all 30 Wan self-attn were replaced and retrained, discarding Wan's compute-heavy
pretraining ("Wan 파라미터를 건드리면 안 될 것"). New design, small-compute:
- Start from the released **ReCamMaster step20000** checkpoint; FREEZE everything
  pretrained (Wan + their fine-tuned self_attn / cam_encoder / projector).
- Replace only ReCamMaster's newly-introduced mechanism (cross-video concat
  attention) with TTT: attention becomes per-video (frozen weights), and a NEW
  trainable TTT fast-weight branch (zero-init output, q/k/v warm-started from the
  block's self_attn) is the only src->tgt channel: update on 7 src chunks (3 latent
  frames each), apply on the tgt half. Trained in THEIR pipeline (envs/recam,
  diffsynth), their loss (flow matching, MSE on tgt half), lr 1e-4 adapters-only.
- Ablation (paper table): {base, in, h, both} — rotary site 2x2, **fixed ladders
  only** (no learnable freqs; user: "논문에는 learnable 안쓸 거"), Plucker phases
  relative to src frame0 (ReCamMaster's own camera convention), gain 1.0, frac 0.5.
- Budget: ~12h x 1 B200 per run, 4 runs parallel (gpus 0-3). Deterministic shared
  data stream + per-step seeded noise/timestep -> paired per-step analysis.
- Eval: val loss on the 64-pair holdout (deterministic) + their 50-step sampler on
  8 pairs vs the ReCamMaster external anchor (gen_recam_anchor, running).
- Infra: latents precomputed once (4066 videos, non-tiled VAE, bf16) to
  datasets/mcv_latents_recam/ (6 shards, gpus 1-6, 2026-07-12 morning);
  save_every 250 checkpointing + auto-resume (user rule).

## Q15. Input-only PRoPE / GTA ports (NVS)  [QUEUED 2026-07-15, user request]
Motivation: a coworker reports PRoPE-in-TTT works well in a ViTTT-based NVS setup
(their setting: ViTTT structure, NO Muon, gate-only 1-layer MLP inner model =
SwiGLU minus W2, PRoPE applied as-is to the TTT path, baseline = ViTTT-for-NVS).
Our F3 negative (prope_ttt 21.852, -0.118) was the FULL q/k/v/o port inside the
LaCT stack (qk L2-norm + weight-norm + Muon), where projective norm distortion is
punished. Missing cells: input-only ports, which is what PRoPE originally is.
- Runs (standard protocol, seed 95, paired vs baseline 21.970 and qk_rope_cam +0.41):
  (a) cam_prope_in: P = lift(K)E_w2c applied to fast q/k ONLY (q <- rep(P^T)q,
      k <- rep(P^-1)k; v/o untouched), at the same site as our input rotary.
      Decision: transform BEFORE the qk L2-norm (document; the norm partially
      breaks projective cancellation either way — that is part of what we test).
  (b) cam_gta_in: same but with the rigid-transform rep (rotation+translation
      homogeneous 4x4, no intrinsics lift) — closer-to-orthogonal control.
- Reading: if (a) recovers to ~baseline or better, F3 must be scoped to the full
  port / value path; if (a) still loses while (b) is neutral and qk_rope_cam wins,
  the orthogonality ladder (projective < rigid < orthogonal rotary) is confirmed
  at matched site and budget.
- (1-layer-MLP coworker replication DROPPED per user 2026-07-15 — not our interest;
  the two input-only ports on OUR stack answer the scoping question.)
- Implementation: reuse the existing prope_ttt rep machinery in lact_ttt_cam.py
  (restrict application points); ~2 new cam_modes + configs. 2 runs x 1.6h.
- Schedule: after Q14 wave-1 (LLM) frees a GPU, or on the video-12k GPUs when done.

## Q20. AdaRoPE-style PER-HEAD learnable frequencies (LLM + all tasks)  [QUEUED 2026-07-16, user request + paper]
Paper: "AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally" (Tsinghua/Tencent,
PMLR 2026; PDF in claude_portable/config/uploads/.../AdaRoPE.pdf). Learnable RoPE that WORKS
on transformer LLMs (up to 2.7B, FineWeb-100B): beats RoPE/PRoPE/ALiBi/NoPE on 7 NLU tasks
(+0.91 avg) and pretrain loss (-0.016).
METHOD (two parts): (1) AdaFreq — learn a log-frequency xi_f^(h) in R PER HEAD, per
dim-pair; map theta_f^(h) = exp(xi_f^(h)); store fp32; INIT from the standard geometric
schedule then learn. (2) AdaScale — per-head length-aware attention temperature lambda(L)
(counteracts attention dilution; less relevant to our fixed-4096 / TTT-no-softmax setting).
WHY IT MAY UNLOCK OUR LEARNABLE FAILURES (F20/F33): AdaRoPE's OWN ablations reproduce our
negatives and show the fix —
  - "Share Freq" (all heads share one learnable schedule) DEGRADES (51.23 -> 50.90 avg;
    loss 2.663 -> 2.685) == our F33 shared-learnable seed-lottery. Sharing is wrong.
  - "Learned Base" (learn a single base b, recover theta=b^(-2f/d)) DEGRADES (-> 50.77)
    == our gain/theta knobs (whole-ladder scale). Wrong parameterization.
  - WINNER: per-dim-pair log-freq, learned directly, PER HEAD. We never tried per-head:
    our h_inv_freq is per-LAYER or globally shared, broadcast across all fast-weight heads.
TRANSPLANT PLAN:
  - AdaFreq on the TTT HIDDEN ladder: h_inv_freq -> [num_fw_heads, P_h] learnable log-freq
    (xi, theta=exp(xi)), fp32, init from the current fixed ladder (gentle g0.1 or standard
    g1.0 per the F35 band rule for the setting). Kernel already takes per-token hcos/hsin;
    make them per-fw-head [num_fw_heads, P_h, S]. This is the one untested learnable axis.
  - Optionally AdaFreq on the input fast-q/k rope too (harder: fla RotaryEmbedding; would
    need a manual per-head rotary like the band-split fast_rotary).
  - Skip AdaScale initially (TTT has no softmax temperature; revisit if a fast-weight
    analogue is motivated).
  - Grid (w128 default per user 2026-07-16): {rope, honly-perhead-adafreq, hpra-perhead-
    adafreq} x seeds; controls = our existing shared/per-layer learnable (F33) and fixed.
  - Deeper-layer freq drift (paper Fig 3-4: early layers lower freq, deep layers higher):
    our per-layer h_inv_freq already allows this; per-head adds the within-layer axis.
Implementation: extend ttt_learnable_freqs to a per-head parameter [num_fw_heads, P_h] with
xi=log parameterization; ~1 kernel-shape change (broadcast hcos/hsin over fw heads). Sanity:
reduces to current when per-head params tied; fp32; init-from-fixed exact at step 0.
Cost: 3B runs ~5-6h each at w128. Read AdaRoPE appendix B/G for exact init + lr before impl.
[WAVE-1 DONE 2026-07-17: honly-perhead 18.61 (s42) / 18.58 (s137) vs fixed honly-g1.0
18.55/18.56; hpra-perhead 18.64 = fixed hpra. Per-head learnable does NOT beat fixed —
consistent with F33's learnable-ladder wall. No wave 2.]

## Q22. Chunk-quantized rotary phases ("block-relative" chunkq)  [CLOSED 2026-07-17 — FAILED: surgeries hidden +0.15 / input +7.9 (both-site dead); trained s42 cell 18.605 vs per-token honly 18.549. Intra-chunk fine phases carry real value; see LEDGER.]
From the 4-agent design round (user request 2026-07-16): proposed INDEPENDENTLY by two
agents (content-position "staircase", attention-transplant "block-relative h-PRA") —
the strongest convergence of the round.
MECHANISM: quantize rotary positions to the update-chunk start, pos -> C*floor(t/C)
(C = lact_chunk_size 1024), hidden site and/or input site. The memory is apply-then-
update at chunk granularity: a probe only ever retrieves from PREVIOUS chunks, so the
finest offset the fast weight can resolve is the chunk index. Per-token phases inside a
chunk only scramble content addressing (the tax the honly GA fought); chunkq removes ALL
intra-chunk scrambling while keeping the full cross-chunk ladder (recency envelope from
slow pairs + chunk tags from fast pairs). Zero new parameters (no F33 lottery), pure
rotation (G-safe), reduces to baseline at gain 0.
P0 KILL-SWITCH (eval-only, running 2026-07-17 gpu0, surgery_chunkq.py): quantize phases
on the TRAINED q17 checkpoints. hidden1024 on honly-g1.0: if cost > ~1/3 of the
honly-NoPE gap (>0.09 ppl), fine phases are load-bearing -> kill before any training.
input1024 on rope: F27b-style measurement of WHERE input rope's +0.20 lives (also gates
the both-site variant). input1 = manual-path sanity (must match base).
TRAINING GRID (gated on P0): honly-chunkq-g1.0 w128 x3 seeds; hpra-chunkq (rope +
chunkq-hidden) x3 seeds if honly passes s42. Flags implemented: ttt_hrope_chunkq /
ttt_input_chunkq (layer_lact_swiglu.py; config threaded).
PREDICTIONS: honly-chunkq <= honly per-token on each seed (tax removal); copy@2560
should DEGRADE vs per-token honly (no intra-chunk resolution) — mechanism check.
Cost: 15 min surgeries + 3-6 x 5.5h.

## Q23. VaPE: value-path rotary — position annotates the PAYLOAD, not the address  [CLOSED 2026-07-17 — PARITY: rope+VaPE 18.608 vs rope 18.609 s42; surgery C=+0.0025 loss (tag partially used, zero net value). One cell, no seeds; impl validated in-tree. See LEDGER.]
Convergent theme #2 (mechanism-first "VaPE" per-token; attention-transplant "vo-chunk"
chunk-level). The ONLY design class that adds information PROVABLY absent from every
failed variant: all failures put a second copy of Delta-t inside the addressing inner
product (findings D/E: redundant, ignored); VaPE leaves addressing bit-identical to
baseline and phase-tags the retrieved payload with its age.
MECHANISM: update on rotated target R_t v_t (dw1 and dhidden follow exactly);
apply o_s = R_s^{-1} f_W(q_s), delta-only variant o_s = w1_init h + R_s^{-1}(w1-w1_init) h
(removes the F27b absolute-phase tax on the init readout). Retrieval channel becomes
sum_t lr_t <h_t,h_s> R_{t-s} v_t — content selects, relative phase disambiguates WHICH
occurrence (recency-resolved induction over superposed matches is a linear readout for
o_proj). Fixed ladder on frac 0.5 of value dims (escape half), gain sweep {1.0, 0.1}.
EVADES: D/E by construction (not an address code); F33 (fixed freqs); G (v is not in
the address path — q/k/h norms untouched); F27b tax (delta-only apply).
KNOWN COUNTER-EVIDENCE: F4 NVS vo_rel -0.12 — but that was dense appearance payloads
with no "when was this written" question; at w128 the 1D memory is capacity-stressed
and recency-disambiguation is real work. One cell first.
KILL CRITERIA: s42 rope+VaPE < rope 18.61 by >= 0.05 else dead; F27b surgery on the
trained model must show the counter-rotation is USED (cost > 0.05 when ablated).
PREDICTION (dissociation): VaPE-only must NOT solve copy@2560 (it changes no addressing).
IMPL: clone prenorm_block_causal_lact_swiglu_hidden_rope -> *_value_rope in
ttt_operation.py (~40 lines; apply_rotary_cols works on v as-is, column-major [b,d,l]);
vi_rot in dhidden + dw1; apply-side inverse via (hcos, -hsin). Autograd-check at phase 0.
Cost: 0.5 day impl + 4 x 5.5h sweep before seeds.

## Q24. Training-dynamics interventions: break the ignore-the-hidden-copy equilibrium  [DEAD 2026-07-17 by commitment probe: (a) dropout C(t) never left the ignore signature even at p~0.5; (b) hidden-first C=-0.0007 at 4k with NO input rope in the model, killed at 21.2k per pre-registered rule. The ignore-equilibrium is the data's optimum, not an init artifact. C(t) tables in LEDGER; q24a ppl endpoint pending.]
Training-dynamics lens: F27b's "model ignores the hidden rotation" is an ATTRACTOR
chosen in the first ~4k steps (finding F); these interventions perturb the acquisition
order, then ANNEAL AWAY (final model = standard fixed-ladder hpra, no inference change).
(a) INPUT-ROPE DROPOUT (top pick ~25%): during training, zero the input-rope phases on
    a random fraction of sequences, p 0.5 -> 0 annealed over the first 30k steps.
    Batches without input rope make hidden the ONLY relative code -> ignoring it costs
    loss; by anneal end the model keeps both. Stateless Bernoulli keyed on
    (data_seed, step, micro) for exact resume.
(b) HIDDEN-FIRST CURRICULUM (~20%): input-rope scale held 0 for the first 4k steps
    (the commitment window), cosine-ramp to 1 by 12k.
INSTRUMENT (both): commitment probe C(t) = val-loss cost of zeroing the hidden rotation
at each checkpoint (F27b surgery, automated). Success = C(final) > 0.1 (vs 0.003
baseline) AND ppl <= rope; C(t) collapsing to 0 mid-anneal = early kill.
EVADES: no architecture change at all — orthogonal to every failed axis; composable
with Q22/Q23 winners.
IMPL: ttt_input_scale buffer + schedule in train_small.py; probe script exists
(surgery pattern). Cost: 2 x 5.5h (a,b at s42) + probes.

## Q27. NWM ego-pose world model  [CLOSED 2026-08-04, user decision — redundant with NVS+CCV+3D-recon, and the 4-frame-history flaw made the adapter inert (base-vs-both val loss 5e-6 over 5500 steps). Stage-0 repro SUCCEEDED and is kept: cdit_xl @4s LPIPS 0.3106 vs published 0.296 (and 0.312 for the num_goals=1 row the eval script actually uses). Assets kept at nwm_ttt/, datasets/nwm_{data,ckpt}.]
Baseline: NWM (Meta, CVPR 2025). Conditioning IS relative ego pose: ActionEmbedder(dx, dy, dyaw)
+ time-shift, injected by adaLN; CDiT block = [self-attn over target tokens] -> [CROSS-ATTN from
target to the 4 context frames] -> [MLP]. The cross-attn is the retrieve-from-past channel.
Assets ready: ckpts cdit_s (49.2M) / cdit_xl (1012M, EMA) at datasets/nwm_ckpt; data at
datasets/nwm_data/processed = recon 11,835 (100% split coverage) / sacson 2,955 / go_stanford 3,696.
Input transform is center-crop to 224x224, so our 640x480 source is not a confound.
STAGE 0 (running): env, config plumbing, forward smoke, reproduce RECON eval vs published
(RECON @4s: LPIPS 0.296 / DreamSim 0.091 / PSNR 15.33).
STAGE 1 design (fixed now, implement after Stage 0 lands) — mirrors the CCV recam_ttt recipe that
beat the ReCamMaster anchor:
  - FREEZE every pretrained weight. Add a zero-init-gated TTT fast-weight branch alongside the
    cross-attention in each CDiT block, so training starts exactly at the frozen model.
  - The TTT branch ingests a LONG history (>> NWM's 4-frame context): update fast weights on past
    frame tokens, apply at the target tokens. It is then the ONLY long-horizon channel, which is
    what makes the ablation interpretable.
  - Rotary coordinates = relative SE(2) pose between the stored frame and the query frame
    (dx, dy, dyaw; optionally + rel_t), the same quantity NWM already conditions on — analogous to
    the 6-D Plucker coords in NVS.
  - 4-cell ablation, seed-matched: base (adapter, no rotary) / in (input site on fast q/k) /
    h (hidden site) / both. Fixed ladders only (learnable is negative everywhere: F33/Q20/Q21/F37).
  - Eval: NWM's own scripts (LPIPS/DreamSim/FID at 1/2/4/8/16 s rollouts) on RECON + the
    known-environment planning eval; report seed-matched paired deltas, not absolute vs the paper.
Scope note: SACSoN public release is 160x120 vs the paper's private 320x240, and TartanDrive/SCAND
were deliberately skipped (no revisit structure / 1.94 TB). We do NOT reproduce the paper's table;
the claim is an internal site ablation on a frozen backbone.

## Q29. CLRS-Text address-dimensionality grid  [READY TO LAUNCH 2026-08-04, node3]

The first CONTROLLED test of F20's dimensionality boundary inside one task. Every
existing datum for "the hidden site needs a multi-D address" is a comparison
ACROSS unlike tasks (1D language/DNA/music null vs 6D NVS/ccv win); Q29 makes it
a within-task measurement.

Host: CLRS-Text (DeepMind, smcleish/CLRS-Text-*, Apache-2.0). An n x n adjacency
matrix is serialized with NO row/col indices and every entry the character 0/1,
so thousands of identical tokens can only be addressed by (row, col). At n=41 the
question is ~3,644 chars vs window 128 -> memory-exclusive.

THE ABLATION IS ON THE COORDINATE, NOT THE DATA. Same problems, tokens, length and
memory load in every arm; only the address changes:
  --clrs_coord_mode 2d -> parsed (row, col)
  --clrs_coord_mode 1d -> (t, t), which recombines the split ladder into
     inv_freq*t = EXACTLY the stock rotary (verified 0.000e+00, sanity mode a)
This is why the earlier data-side plan was abandoned: CLRS array algorithms are
1-D only in the QUESTION (a_outer_max: bfs n=41 -> 4, insertion_sort n=25 -> 25),
so a data-side 1-D control inverts the contrast in the loss region and confounds
dimensionality with locality.

Grid: 7 cells = base + {in, h, both} x {2d, 1d}. base has no rotary so its
coordinate is provably inert (0.000e+00) -- one run, not two.
Launch: lact_llm/run_clrs_grid.sh 0,1,2,3   (~1.2B tokens/cell, 2 waves on 4 GPUs)
Prompt for the node3 session: NODE3_Q29.md

PRE-REGISTERED: the h-over-in increment appears in the 2d arms and vanishes in the
1d arms. Equal gains in both refute the dimensionality hypothesis, and we say so.

Built + verified on node1 (2026-08-04):
- clrs_data.py selftest 13/13 (incl. real samples: bfs n=41 sweeps outer to 40,
  insertion_sort stays at 1)
- sanity_clrs_coords.py 11/11: 1d == stock rotary 0.000e+00 at BOTH sites; 2d
  changes the loss; rotary-off makes the address inert 0.000e+00; per-sequence
  addressing; finite backward
- infer_n() (train repo has no length column) vs test repo's length: 37,144/40,000,
  and 100% on all five algorithms used
- corpus: 90,000 blocks x 4096 (1.47 GB), 0 oversize dropped; val 512 blocks from
  the held-out test seeds; supervised span verified to be exactly the answer + EOS

## Q28. Cross-task #3: 3D reconstruction — TTT-RoPE on tttLRM  [PORT IN PROGRESS 2026-08-04]
Host: **tttLRM** (CVPR 2026, github.com/cwchenwang/tttLRM). Chosen after FSM (posed NVS = our own
task) and ZipMap (no pose-conditional input mode -> feeding GT poses would change the model's
identity and confuse reviewers) were both rejected.
TASK: posed images (12ch = ray_o|ray_d|ray_o x ray_d|RGB) -> **pixel-aligned 3D GAUSSIANS**,
rasterized by gsplat at target poses. The differentiator vs our NVS work is that the fast weight's
apply output decodes to 3D GEOMETRY, not pixels — plus ~250x longer context (64+64 views x 8,040
tokens = 514,560 tokens in one fast weight, vs ~2k in lact_nvs) and an autoregressive variant.
Honest caveat: it is initialized from TTT-LVSM weights, i.e. the same LaCT-LVSM trunk our work
builds on, so it is the same task FAMILY. User accepted this on the Gaussian-output grounds.
MEMORY (verified in the loaded checkpoint): genuine SwiGLU fast weight, 24 blocks,
w0(1,768,3072)/w1(1,3072,768)/w2(1,768,3072), num_heads=1, inter_multi 4. Sites: qk after
lact_ttt.py:317; hidden at :163 forward / :165 inverse / :239 apply.
NEW TRAP (worse than FSM/ZipMap): TTTOperator has FIVE fields (start,end,fast_weight,update,apply),
so the single apply expression is reachable in THREE fast-weight states, and the shipped op
builders exercise only two — a bug in the third would pass every stock run.
DATA: local DL3DV (dl3dv_undistorted_960), train filtered to **9,995** scenes (130 of the 140 test
scenes also live in train — verified twice with comm -12), test 140. The scene model is DL3DV-ONLY
(Objaverse is the separate object model), so a DL3DV-only fine-tune costs nothing.
ANCHOR: their Table 2 on DL3DV-140 — 32 views 25.07/0.822/0.215, 64 views 25.95/0.844/0.195.
CELLS (4, not 5): camera is already injected as content, so base == the content arm.
base / +in / +h / +both, same seed/data order/budget, from dl3dv_full.pt at lr 1e-5.
Optional 5th: zero the 9 ray channels and keep ONLY the rotary — the analogue of our LLM nope arm,
testing whether addressing can REPLACE content (large distribution shift; not a matched arm).
COST (measured on 1x B200, linear at 0.147 s/view): 16+16 views, grad-accum 8, 1000 opt steps =>
~13 h for all four cells together (one per GPU, concurrent). Licence: Adobe Research,
noncommercial research only — fine for the paper, flag in acknowledgements.

## Q26. GbR single-branch input rope — BREAKTHROUGH CANDIDATE  [ACTIVE 2026-07-17]
From the Q25b diagnostic: at w128/3B/s42, gate-only rope 18.471 and content-only 18.502
BOTH beat standard both-branch rope 18.609 (nope 18.81). Finding: rotating both SwiGLU
input branches is a ~0.11-0.14 tax (the multiplicative interaction compounds the
rotation and scrambles the content channel); one rotated branch = full relative code +
position-clean content channel. Gate-only earns +0.34 over NoPE net.
IN FLIGHT: gate s137/s211 (gpu2/6), content s137/s211 (gpu0/1), copy-dissociation
q25b_copy_gate (gpu4; if gate-only still solves copy@2560, the clean-content split
costs nothing on position-addressed retrieval either).
IF 3 SEEDS HOLD (the 1D goal cell "beats input rope" is then MET by an input-site
redesign): (a) gate+hidden stacking (kernel work: branch_rope + hcos/hsin); (b) NVS
port (cam_mode flag in lact_ttt_cam.py — the cam rotary also rotates both branches);
(c) CCV port; (d) w1024 tax-invariance check.

## Q25. Lower-priority design-round items  [QUEUED 2026-07-17, run if Q22-Q24 die]
(a) Chunk precession of the w1 delta: after each chunk update, right-rotate the
    accumulated delta (w1-w1_init) by a fixed per-chunk angle ladder — stored addresses
    precess with age (recency kernel in the STATE; the norm-exact "forgetting" the
    constraint set allows). ~15 lines in ttt_operation.py. Risk: still a Delta-chunk
    address code (D applies); only 3 offsets at 4096/1024.
(b) GbR gate-branch-only input rope: rope only the w0(silu-gate) q/k copy, content
    branch (w2->w1) unrotated. 2x2 {gate-only, content-only} decomposes WHERE finding
    A's +0.20 lives — mechanism knowledge even on a loss.
(c) Conjugate-paired ladder: h_inv = cat([f, -f]) — gives the model an even/odd basis
    to keep the recency envelope and drop the offset code. Cheapest (~5 lines); 4k-step
    partner-correlation check falsifies for free.
NOT ADOPTED from the round: delta-decay gamma on the accumulated delta (agent's own
top pick, but it is a scalar forgetting gate, not a positional embedding — fails the
"reduces to baseline at zero phase" rule and the DeltaNet-adjacency makes it a framing
fight); copy-data curriculum (changes training data distribution, weakest expected
transfer); asymmetric quantization (breaks shift-relativity).

## Q13. Shared learnable frequency across layers (NVS)  [DONE 2026-07-17 -> F37: sharedf 22.338 vs per-layer 22.420, t=-12.5 — sharing strictly hurts; no wave 2. LLM extension subsumed by F33 (shared-ladder lottery).]
User: per-layer learnable gains may be the reason learnable ladders lose to fixed
(each of 6 layers gets its own 6xF gain -> too much freedom). Test ONE learnable gain
tensor shared by every layer ('sharedf' cam_mode flag, commit pending: class-level
registry; named_parameters dedupes so the optimizer sees it once).
- Wave 1 (paired, seed 95, standard protocol): cam_qk_rope_cam (per-layer learnable,
  fresh rerun) vs cam_qkrope_sharedf (shared learnable). Reference points: fixed-ladder
  pra_hi 22.389 (F25), old-env per-layer learnable qk_rope_cam 22.375.
- Auto-launches on gpus 4/5 when the Q12 wave-1 g02/g005 runs finish (~evening).
- If shared > per-layer: seeds 137/211 + hidden-site shared (gain_h) follow.
- **Q13-LLM extension (user 2026-07-14)**: ttt_sharedf in lact_llm — h_inv_freq (hidden
  ladder, init = gentle gain-0.1 champion) + fwqk_dfreq (input additive deltas) shared
  across all 12 layers. Pair (3B ds42, rope ON): q13L_stack_lf_perlayer (control; the
  regime where learnable degenerated in F20) vs q13L_stack_lf_shared. References:
  rope 18.405 (fixed), hpra-g0.1 18.415 (fixed gentle). Doubles as a Q12 wave-2
  candidate: a shared learnable ladder starting at the champion spectrum. Auto-launch
  queued (gpu6 after Q12 f25_g02; gpu4 after the Q13-NVS pair).

## Q10. Video revisit with Q9-informed variants  [CANCELLED 2026-07-12 — superseded by Q11 (user pivot); g03/g01 killed at step ~650. The 4-anchor grid + evals remain valid as the "replace-everything" data point: F30/F30b + generation metrics @13999]
User: F21/F22 neutrality was measured with the PLAIN hidden ladder; the Q9 discovery
(low-gain slow ladder turns hidden-1D from coin-flip/negative into a draw-robust WIN)
may transfer to video — and video's spacetime coordinate is multi-dimensional, where the
theory says the hidden address space has more relative structure to exploit.
- Plan: port the Q9 genes (hrope gain / frac / delta_only) to lact_ar_video's hidden-rope
  path; then paired per-step-loss runs (deterministic noise, IMPL_SPEC_CCV protocol):
  h_pra_gain003 / h_pra_gain01 (and optionally delta_only) vs base, 60-step sanity first.
- GPUs: after the current ccv 20k grid (gpu2-5) finishes (~2026-07-11 evening); the grid
  also provides fresh base/pra/both/pra_fixed anchors to compare against.
- Choose gains informed by Q9's final bracket + the video ladder's own units (phase per
  latent-frame/pixel, not per token — recompute the "top-frequency period" equivalent).
- PORT DONE 2026-07-10 (commit 07e6f3d): genes in ar_lact_swa_repeat.py, defaults
  bit-identical (T0-T3). Ladder analysis: in ccv the grid-carrier phase CANCELS across
  src->tgt recall (matched grid indices) — only the Plucker mode carries cross-view
  offsets, so the sweep targets the Plucker hidden gain. At default gain 1.0 the top
  Plucker frequency wraps ~1.6 turns over the measured recall delta (RMS 0.209/coord) —
  the same "scrambling" regime that lost in 1D. LLM 0.1/0.03 equivalents: video Plucker
  hidden gain ~0.3 and ~0.1.
- VARIANT SET (launch when gpu2-5 free): abl_ccv_both with ttt_hrope_gain 0.3 and 0.1
  (2 runs, paired vs the fresh base/pra/both anchors), 20k steps, deterministic noise.

## Q8. LLM ds43 early-read pair (F27b confirmation)  [RUNNING 2026-07-09, gpu1 sequential]
rope vs hpra at data_seed 43 (different data draw, SAME seed-42 val cache), 0.5B tokens
(~15.2k steps, ~2h each). Reading: is the hidden-1D deficit sign stable across data draws
within the rebuilt env? (seed-42 reference at 15k: hpra−rope = +0.99 ppl, shrinking to +0.23
by 91.5k). If ds43 also positive -> new-env result draw-robust; old-env F19 remains a
different-draw outlier. If ~0/negative -> confirms draw sensitivity of a ~1% effect.
Requested 2026-07-09 (user). Completes the F19 matrix with the missing hidden-only
cell, rerun as a FULL 2x2 grid in the rebuilt env (torch 2.9.1+cu130 + new fla; old
runs' env is gone, so mixing old/new numbers would confound):
- abl_nope  {ttt_nope}                    - abl_rope  {} (stock fw-RoPE)
- abl_hpra  {ttt_hidden_rope}             - abl_honly {ttt_nope, ttt_hidden_rope}  <- NEW CELL
Protocol: F19 recipe — 200M LaCT LM (768/12L), 3B tokens fineweb-edu streaming,
fixed data order (data_seed 42), bs 8 x 4096 -> ~91.5k steps, val ppl on the fixed
2M-token cache. ~12h/run. Reading: does the hidden channel work WITHOUT the input
channel in 1D (mirrors NVS F25 where hidden-only carries +0.96 of +1.08)?

## 2026-08-11 queued: pv_h_f85 (Q35 coverage control, user-approved)
- **What**: scratch_pv_h with num_freqs_h 42 -> 85 (hidden ladder 49% -> 99.6% of
  d_h=1024; density 2.71x -> 5.48x). Everything else identical to pv_h.
- **Why**: pv's h flip (h - base significantly negative at 20k/25k) is confounded --
  the d256 shrink halved hidden coverage vs both F45 (100%) and NVS h_pra_hi (98.4%).
  F85 restores full coverage at the cost of density, separating (view regime +
  crowding) from (coverage) as the cause.
- **How**: watch_launch_f85.sh auto-launches `run_scratch_pv.sh h_f85 1` when pv_in
  writes final.pt (~05:30). 30k steps, ~1.3 days -> done ~Wed mid-day. Read against
  pv_base AND pv_h at v32/140.

## 2026-08-12 queued: CCV sampled-metrics eval (user-approved)
- **What**: for each F54 cell (ccv_base / site_in / site_hidden / site_both, EMA at
  the common step), SAMPLE the same 64 held-out val pairs with fixed seeds and score
  generated frames vs MultiCamVideo GT: PSNR / SSIM / LPIPS, paired per pair.
  Fills the "evaluation metric" gap in the paper's CCV table (currently val loss only).
- **When/where**: after pv_h_f85 completes (GPUs 2/3 free; GPUs 0/1 go to the
  dl3dv32 GTA/PRoPE baselines). ~4-8 GPU-hours.
- **Status**: pipeline recon in progress (checkpoint paths, 64-pair list, sampling
  entry point); runner script to follow.

## 2026-08-12 queued: Q44 -- can parameterization rescue ray-rotary at wide baselines?
User hypothesis (after RayRoPE arXiv 2601.15275 Table 1: RoPE-on-rays ~ group actions
on 2-view Objaverse): our wide-baseline rotary failures may be parameterization, not
physics. Two stages:
- **Q44a (attribution, transformer)**: Q37 plucker_rope (gobj 13.9 vs GTA/PRoPE 20.7)
  rerun with ONE lever each: `plucker_rope_rel` (first-view-identity norm),
  `plucker_rope_tail` (p-RoPE half coverage, lowest rungs + content tail),
  `plucker_rope_gentle` (band x 1/8). Implemented in prope_run/nvs/lvsm.py; runner
  run_prope_repro.sh takes the new arm names. ~1 h/cell on B200. Which lever closes
  the 7 dB gap = the prescription.
- **Q44b (transfer, TTT)**: port the winning combo to qk_rope_cam on gObjaverse
  (coverage flag + omega_scale exist; first-view-identity needs a pose-norm variant).
  Success = input rope beats base (was -0.41); stretch = TTT-RoPE(+transport) beats
  imgvo 22.58.
Mechanism note: distance sets phase magnitude, but damage requires wrap x collisions;
2-view windows have no collision partner (F53's NN-angle statistic, F57's monotone
band curve are the internal evidence).

## Q44a EXTENSION (2026-08-12 22:20, after reading RayRoPE's public code)
Their Table-1 "RoPE on rays" baseline (pos_enc/rope_global_ray.py) is NOT q/k-only
rotary: apply_fn_kv rotates V too and apply_fn_o counter-rotates the output --
phase-form v/o TRANSPORT (our F59/F60 carrier ingredient), plus period-calibrated
gentle bands (top omega ~1.5pi vs our 16pi), bounded coords (center+directions, no
Plucker moment), first-view-identity norm, 100% coverage (tail is NOT their trick).
Two cells added behind the first wave: `plucker_rope_vo` (transport only) and
`plucker_rope_combo` (rel+gentle+vo = their baseline's analog on our coords).
PRE-REGISTERED: the slot account predicts vo is the LARGEST single lever, tail the
smallest; combo should approach GTA/PRoPE (20.7-20.8).

## Q44a waves 2-4 ledger (2026-08-13 morning)
Wave 2-3 results (train PSNR @80k; plain plucker_rope 13.9, GTA/PRoPE 20.7-20.8):
vo 12.6 / tv 12.6 / rtv 13.6 / combo 14.9 -- EVERY transport-bearing cell is worse
than its transport-free twin (tail 16.2 -> tail+vo 12.6). Phase transport through
the HOT ladder scrambles the carrier: F60's "wrapped ladder eats the transport
gain" reproduced at the attention site. Pre-registered "vo largest lever" is
REFUTED for the hot-ladder parameterization; rel(+2.0)/tail(+2.3) stand.
Wave 4 (launched): `rt` = rel+tail (best transport-free combo), `raycal` =
faithful RayRoPE-baseline port (center+direction coords, per-coordinate calibrated
2-rung bands tiled to 48 dims, first-view-identity, their v/o wiring). raycal
closing to ~20 = recipe transfers (calibration+coords+transport was the whole
story); raycal stalling = their Objaverse gain needs their 2-view protocol/data.

## Q44a wave 5 (user goal restated 2026-08-13: REPRODUCE RoPE-on-rays ~ PRoPE on our gObjaverse)
Attribution correction: our prope testbed trains ref_views=2 (prope default), SAME
view count as RayRoPE's protocol -- the earlier "2-view vs 8-view window" account
is WRONG. Remaining suspects: (a) DATA-level first-view-identity canonicalization
(theirs normalizes poses before raymaps AND PE; our gobj feeds raw world coords,
|center|~1.7, pair angle median 105 deg; my rel lever canonicalized the PE only,
leaving PE and input raymaps in different frames), (b) port fidelity (3 rays/patch,
24 distinct calibrated phases vs my 12 tiled).
- `raycal2` (gpu5): data-level canon at model entry (raymaps+PE consistent, t x0.5
  to fit the p0 period-4 calibration) + rich PE + their transport.
- `raycal3` (gpu6): rich PE + PE-side rel only (no data canon) -- isolates (a) vs (b).
Read: raycal2 ~ 20 and raycal3 low => frame CONSISTENCY was the missing piece.
Both low => their number needs their render distribution (random FOV/distance).

## Q44a wave 6 (2026-08-13, after auditing RayRoPE's TRAINING pipeline)
Two pipeline-level discoveries (RayRoPE/nvs):
1. **ray_encoding=camray is their default**: input raymaps are computed with
   IDENTITY extrinsics (intrinsics-only, camera-frame) -- ZERO pose in the input
   tokens. Pose flows exclusively through the attention encoding. Their Table 1
   is "pose-in-tokens (Plucker raymap row, worst) vs pose-in-attention (all other
   rows)". Every cell of ours so far kept world-frame pose raymaps in the tokens
   and ADDED phases on top: a different experiment (redundant + frame-mismatched
   pose channels).
2. **objaverse_dataset.py normalizes at data level**: normalize_view1_to_identity
   + max-radius scale.
Wave 6 cells: `raycal4` (gpu7) = camray input + data canon + rich calibrated PE +
transport = their architecture end-to-end on our renders; `prope -camray` (gpu4) =
the matched PRoPE bar under the same input encoding. Target: raycal4 within ~0.5
of prope-camray = reproduction achieved.

## 2026-08-31 NEW PROGRAM: camera embedding at wide baseline (gObjaverse) — user refocus
User decision (2026-08-31): concentrate on the camera-embedding (NVS) task; develop the method so it
works on Objaverse/DL3DV, not only RE10K/ReCamMaster. Analysis + 12 hypotheses: `OBJ_ANALYSIS.md`.
Protocol: F51 recipe via `lact_nvs/run_gobj.sh` (gObjaverse orbit renders in /tmp/gobj, 30k, seed 95,
paired vs `gobj_base_s95/eval_v2.json` = baseline re-evaluated on the current test index).
- wave 1 node1 (launched 13:15 KST): oracle_both (H1, GT-depth 3D-point rotary in+hidden; DEPTH_DIR),
  shell_in (H2 input), shell_h (H2 hidden), camray_rotraw (H7 pose-free tokens + rot transport).
  [oracle/shell relaunched 13:40 after a sqrt(0)-backward NaN fix in `_chord_t`]
- wave 1 node2 (NODE2_PROMPT_OBJ.md, awaiting the user's paste): attn_nope / attn_prope (attention
  ceiling controls), hrot_rotraw (H4), imgvo_himg (H10); then camray_properaw, h_dpra42, shell_iso_in.
- wave 2 candidates (implement next): RayGTA per-token ray-frame transport (H6), epi_quad (H9),
  rot_content = gate-invariant/content-relative SwiGLU (H8 stage 1), hgate (H11), oracle_in (H1b).
- reference re-evals on the current test index: outputs/gobj_{base,imgvo,rot_raw,prope_raw,input,
  hidden,prope_imgrope,both}_s95/eval_v2.json (script outputs/_smoke/reeval_refs.sh).
