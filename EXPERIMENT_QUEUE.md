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

## Q11. ReCamMaster-frozen + TTT-adapter 2x2  [ACTIVE 2026-07-12, user decision — supersedes Q10]
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
