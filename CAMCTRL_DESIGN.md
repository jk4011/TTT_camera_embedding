# Camera-Controlled Video Generation with PRA: Experiment Design

Status: DRAFT v1 (2026-07-06). Survey section to be filled from web-research agents.
Goal: test whether Plücker Rotary Addressing (PRA) on the TTT layer differentiates
against the standard feature-injection recipes of camera-controlled video generation,
in the setting where the fast-weight memory is load-bearing.

## 0. Why this task fixes the video boundary condition

Plain AR video finetune was neutral for h-PRA (F21) because the SWA covers the
immediately preceding AR window and adjacent-frame redundancy leaves the TTT memory
almost no exclusive workload. Camera-controlled re-rendering changes the information
flow: the source-camera video is written into the fast weights, and the target-camera
tokens must retrieve appearance/geometry from it across a pose change, at sequence
distances far beyond the SWA window. Cross-camera retrieval is memory-exclusive,
exactly the regime where PRA earned +1.2 dB in NVS (where the TTT layer is the only
cross-view path).

## 1. Task definition (ReCamMaster-style re-rendering)

- Input: source video V_src (cam A, 81 frames, MultiCamVideo) + target camera
  trajectory {c2w_B(t)} (+ fixed text caption, as in current pipeline).
- Output: target video V_tgt (cam B of the same scene, synchronized).
- Advantage of MCV as testbed: GT target video exists for every (scene, cam) pair,
  so we can report per-frame PSNR/LPIPS against GT, not only FVD, and the val split
  ships 10 canonical held-out trajectories (val/10basic_trajectories).

## 2. Architecture and sequence layout

Backbone: Wan2.1-T2V-1.3B + our SWA+LaCT hybrid (lact_ar_video/minVid), attn_every_n=100
(no full-attention layers), fw_head_dim 768, inter_multi 2, weight_norm, qk_l2_norm.

Sequence (per sample):
```
[ SRC clean: 21 latent frames x 1560 tok = 32,760 tok  (write-only chunks) ]
[ TGT interleaved AR as today: n1 c1 n2 c2 ... c6 n7    (denoise + write)   ]
```
- SRC written into fast weights in 7 chunks of 4,680 tokens (or 3-4 bigger chunks;
  ablate later). No denoising loss on SRC.
- TGT processed exactly as the current AR pipeline (loss on noisy windows).
- SWA window stays 4,680: for every TGT token, the SRC content is outside the local
  window => cross-video transfer flows only through the fast weights.
- Memory cost: sequence length doubles (~65k tokens with the repeat structure).
  Full attention over concatenated videos (ReCamMaster) is quadratic in this length;
  our cross-video channel is linear. This is itself a differentiation claim (H3).

## 3. Conditioning variants (the 3-run grid)

The field-standard injection on this exact task/backbone is ReCamMaster's:
per-frame 3x4 extrinsics (relative to the condition camera) flattened to 12-dim,
one Linear(12 -> dim) "cam_encoder" per DiT block + identity-init projector, output
ADDED to block features; source video conditioned by frame-dimension concat through
the backbone's existing attention; trainable = cam_encoder + projector + self-attn.
(ReCamMaster's own ablation: frame-dim concat >> channel-concat raymap >> view-attn,
FID 57.1 vs 74.1 vs 80.5.) Our grid therefore uses cam_encoder as the absolute
baseline, not raymap-concat:

| run | camera injection | TTT layer | question answered |
|---|---|---|---|
| ccv_base | ReCamMaster-style cam_encoder (Linear 12->dim per block, relative extrinsics, identity projector) | pose-blind | industry-standard absolute injection adapted to SWA+LaCT backbone |
| ccv_pra | none (no cam_encoder) | input rotary + hidden rotary, phases = per-token 6D Pluecker rays | pure relative addressing vs pure absolute injection, at ~0 added params |
| ccv_both | cam_encoder | + PRA | are the two mechanisms additive? |

All three use the same frame-dim concat [SRC clean || TGT AR-interleave] sequence,
with cross-video interaction carried by the TTT memory (SWA window < concat gap).

PRA instantiation (transplant of the NVS recipe):
- Per-token ray: token (frame t_lat, py, px) of camera C -> RGB frame index 4*t_lat
  (official ReCamMaster loader samples frames 0,4,...,80 -> 21 latent frames; we use
  the same mapping), per-frame c2w_C(t) from camera_extrinsics.json with the official
  UE->CV conversion (c2w[:, [1,2,0,3]]; c2w[:3,1] *= -1; translation / 100 cm->m),
  intrinsics from focal group (sensor 23.76mm; e.g. f24 => fx=fy~1292.9 px at 1280^2,
  adjusted for the 480x832 resize/crop exactly as lact_nvs resize_and_crop), pixel
  center of the 2x2 latent patch -> ray (o, d), Pluecker pi = (d, o x d).
- Canonical frame: per-scene normalization with mean pose over BOTH cameras' frames
  (port normalize_with_mean_pose; scene scale from camera spread).
- Input rotary on q,k post-L2-norm; hidden rotary over fw head (h_rope_dim 768),
  ladders as in NVS final recipe; omega_map deferred to a follow-up run.
- SRC and TGT tokens both get their own camera's rays; the relative phase between a
  TGT query and a SRC key is exactly the pairwise ray offset. Note the contrast with
  ReCamMaster's relativization: they pick ONE reference (the condition camera) and
  hand the network per-frame absolute features in that gauge; PRA relativizes every
  (writer, reader) pair inside the operator, with no reference choice at all.

## 4. Hypotheses and how each is falsified

- H1 (workload): ccv_pra beats ccv_base on paired val loss and on camera-following
  metrics. Falsified if paired Δloss ≈ 0 as in F21 despite the memory-exclusive path.
- H2 (relative > absolute): ccv_pra generalizes better than ccv_base to the 10
  held-out basic trajectories (val split, never seen in training); additionally a
  reference-gauge perturbation test at inference (re-express all extrinsics in a
  rotated/translated world frame while holding the canonical-frame normalization
  procedure fixed) leaves PRA outputs unchanged by construction, whereas
  cam_encoder features change with the gauge choice.
- H3 (cost): PRA adds ~0 params (phase ladders only) vs adapter/encoder params of
  standard recipes, at linear cross-video cost. Report param/FLops table.

## 5. Metrics

Phase 1 (cheap, during/after training):
- Paired val flow-matching loss on held-out scenes, deterministic noise, fixed
  (scene, srcCam, tgtCam) list => same per-step paired-t protocol as F21.
Phase 2 (generation quality, protocol = ReCamMaster/CamI2V lineage):
- GT-pixel metrics: PSNR / LPIPS / SSIM of generated V_tgt vs the rendered GT
  target-cam video (MCV has GT for every (scene, cam); WebVid-based papers could
  not report this, we can. Strongest and cheapest signal.)
- Camera accuracy: estimate generated-video poses (GLOMAP as in ReCamMaster, or
  VGGT as the modern drop-in), align to GT trajectory, report RotErr / TransErr /
  CamMC with the CamI2V formulas (relative-to-first-frame, scale-normalized).
- Source-target sync: Mat.Pix (GIM matcher) optional; CLIP-V optional.
- External anchor: evaluate the released ReCamMaster-Wan2.1 step20000.ckpt on our
  val pair list with the same metrics (not apples-to-apples training data/steps,
  but a useful published reference on the same backbone+dataset).

## 6. Training protocol

- Data: 3,000 train scenes already on /tmp (29,990 clips, scene-complete: all 10 cams
  per scene => any (src,tgt) pair available). Pairs sampled per scene per step.
- 20k steps, batch 1/GPU, lr 1e-4 (= released ReCamMaster-Wan recipe; they used
  8 GPUs where we use 1/run), save_every 2000,
  deterministic noise ON, wandb disabled, TRITON/INDUCTOR caches on NFS.
- 3 runs on 3 GPUs (~26-30h given 2x sequence length; measure step time first,
  shorten to 12k if >40h). GPU 3 reserved for eval/dev.
- Timing: launch after v20k finishes (~20h remaining) OR kill v20k if user prefers
  immediate start. v20k currently 33% done and tracking "neutral", worth keeping only
  as the paper's long-budget no-harm datapoint.

## 7. Implementation checklist (lact_ar_video)

1. `minVid/data/multicam_pair_dataset.py`: scene index; (src,tgt) sampling; two-video
   load + resize/crop; extrinsics JSON parse (keys frame{i}/cam{XX}, values are 4x4
   matrix strings split on "] ["; UE convention with translation in 4th row, cm);
   official UE->CV conversion (column permute [1,2,0,3], Y-col flip, /100); frame
   subsample 0,4,...,80; intrinsics from focal group; canonical-frame normalization;
   returns frames [2,3,81,H,W] + per-latent-frame c2w pairs + intrinsics.
   (Port the parsing from the official ReCamMaster train_recammaster.py.)
2. Plücker phase builder (port of lact_nvs compute_camera_info to the latent token
   grid; fp32, no_grad, matching NVS conventions).
3. `ar_lact_swa_repeat.py`: accept external per-token phases (replace carrier-trick
   grid phases when camera phases are provided); extend q/k input rotary site (v20k
   only had hidden rotary phases via carrier; input site exists in LaCT LLM code to
   port).
4. Model wrapper: sequence concat [SRC clean || TGT interleave]; update ranges
   (write on SRC chunks + TGT clean chunks; loss mask = TGT noisy only); optional
   ReCamMaster-style cam_encoder per block (Linear 12->dim on per-frame relative
   extrinsics + identity-init projector) for ccv_base / ccv_both; optional
   robustness tricks from the paper (mild noise on SRC latents; 0.2-prob task mixing)
   deferred, keep the first grid minimal.
5. Configs: abl_ccv_{base,pra,praonly}.yaml; eval pair list json (held-out scenes).
6. Sanity: overfit 1 scene x 2 cams for 200 steps; check SRC->TGT copying works
   (loss should collapse vs base without SRC).

## 8. Survey: widely-used camera-control methods (TO FILL from agent reports)

(placeholder: injection-mechanism table, absolute vs relative classification,
evaluation protocols, and the differentiation matrix against PRA)
