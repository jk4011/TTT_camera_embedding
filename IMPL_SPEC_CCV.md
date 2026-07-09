# Implementation Spec: Camera-Controlled Video Gen (ccv) runs

**POST-RESET UPDATE (2026-07-09)** — paths below predate the T_B workspace loss; current truth:
- Python: `$ROOT/.venv_llm/bin/python` (rebuilt: torch 2.9.1+cu130, flash-attn 2.8.3 built
  against system nvcc 13.1; the cu128 torch used by lact_nvs cannot build flash-attn here).
- Wan ckpt: `/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/wan_ckpt/` (not /tmp).
- Data: `/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/MultiCamVideo-Dataset/train` serves as
  BOTH data_root and cam_root (the T_B NFS original is gone; full HF re-download).
- GPU plan: grid of all 4 variants on GPUs 2-5 via `run_ccv_grid.sh` (GPUs 0-1 finish Q4 NVS).
- The implementation checklist below was already completed pre-reset (code survived in git);
  only its environment/paths/GPU notes are stale.

Companion to CAMCTRL_DESIGN.md. Target repo: `lact_ar_video/` (minVid). Python:
`$ROOT/.venv_llm/bin/python` (ROOT = repo root). Wan ckpt at /tmp/wan_ckpt.
Env for any run: TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas,
TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump,
TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm,
TRITON_CACHE_DIR=$ROOT/.cache_triton, TORCHINDUCTOR_CACHE_DIR=$ROOT/.cache_inductor,
HF_HOME=/tmp/hf_cache. /tmp is noexec: run scripts via `bash script.sh`.
GPUs 0,1,2 are BUSY with live v20k training. Use ONLY GPU 3. Never pkill python.

## Sequence design (one sample)

Latent frames: src video -> 21 clean frames; tgt video -> existing AR interleave
[n1 c1 n2 c2 ... c6 n7] (39 latent frames). Full sequence = [SRC 21 || TGT 39]
= 60 latent frames x 1560 tokens = 93,600 tokens. src_prefix_len = 32,760.

- Fast weights: write each SRC chunk (4,680 tok = 3 latent frames, 7 chunks) then
  apply it with post-write weights; then run the EXISTING interleave logic on the
  TGT region with all indices offset by src_prefix_len (first tgt noisy chunk now
  applies with post-src weights instead of initial weights).
- Window attention: src region attends only within src (chunk into 4,680-token
  windows, full attn inside each window; no cross-window needed); tgt region runs
  the existing attention path unchanged. NO attention across the src/tgt boundary
  (deliberate: all cross-video information must flow through fast weights).
- Loss: only tgt noisy chunks (existing loss path, indices offset).
- Base-attention rope positions: src frames get their own t=0..20 grid; tgt keeps
  existing positions (separate attention regions, no conflict).

## Files

### 1. NEW minVid/data/multicam_pair_dataset.py
`MultiCamPairDataModule` mirroring SimpleVideoDataModule's interface (READ
minVid/data/simple_video_dataset.py first; match decode/resize/crop exactly).
- Index: list of (scene_dir, src_cam, tgt_cam), src_cam != tgt_cam, sampled with
  index_seed; num_pairs config. Scene dirs under data_root (/tmp/mcv_clips layout =
  f{XX}_aperture{Y}/scene{N}/videos/cam{XX}.mp4).
- Camera json: /tmp copy may lack cameras/; read camera_extrinsics.json from the
  NFS original (/NHNHOME/WORKSPACE/26msit001_T_B/POSTECH-CGLAB/dataset/MultiCamVideo-Dataset/train/<same relpath>/cameras/).
- Parse: data["frame{i}"]["cam{XX}"] = "[a b c d] [e f g h] [i j k l] [m n o p]"
  -> 4x4 rows; translation sits in the 4th ROW (row-vector convention) -> transpose
  to column-convention c2w. Then official UE->CV conversion:
  c2w = c2w[:, [1,2,0,3]]; c2w[:3,1] *= -1; c2w[:3,3] /= 100.
  Subsample frames 0,4,...,80 -> 21 poses per cam.
- Canonicalization: port normalize_with_mean_pose + scene scale normalization from
  lact_nvs/data.py (READ it), applied jointly over the 42 poses (both cams).
- Intrinsics: focal from dir name (18/24/35/50mm), sensor 23.76mm, source 1280x1280:
  fx=fy=focal/23.76*1280, cx=cy=640. Apply the SAME resize+crop math as the frame
  decode path to get K for 480x832 (expected: scale=832/1280=0.65, resized 832x832,
  top crop 176, left 0 -> fx'=fx*0.65, cx'=416, cy'=416-176=240; verify against the
  actual decode code and keep pixels/K consistent).
- Returns per item: frames_src, frames_tgt [3,81,480,832] in [0,1], c2w_src,
  c2w_tgt [21,4,4] fp32 canonical, K [3,3], caption (fixed string as today).
- Sanity inside the module (run once in __main__): after canonicalization camera
  centers are O(1); ray directions unit; most cameras look toward the origin
  (dot(view_dir, -normalize(center)) > 0 for >80% of poses).

### 2. NEW minVid/models/blocks/cam_phase_builder.py
- plucker_per_token(c2w [21,4,4], K) -> [21, 1560, 6] fp32: latent grid 30x52,
  token center pixel (u,v)=((px+0.5)*16, (py+0.5)*16); d = normalize(R_c2w @ K^-1
  [u,v,1]); o = c2w[:3,3]; pi = (d, o x d). Token order must match the
  transformer's patchify order (READ the patchify in the Wan wrapper).
- Ladders: copy frequency conventions from lact_nvs/lact_ttt_cam.py (qk_rope_cam
  input ladder, h_pra hidden ladder; READ that file): per Plucker coord F
  geometric frequencies. Input rotary: fill the TTT q/k per-fw-head dim (768:
  6 coords x F=63 -> 378 pairs = 756 dims, rest untouched). Hidden rotary: reuse
  h_rope_dim machinery (768 dims -> 384 pairs = 6 x 64). Produce (cos,sin) tables
  [L_tokens, dims] fp32 under no_grad except learnable-freq params.

### 3. MODIFY minVid/models/blocks/ar_lact_swa_repeat.py
- Kernel `ar_fast_weight_..._hidden_rope` and the plain one: new arg
  src_prefix_len=0 implementing the prefix write+apply loop, then existing logic
  offset. Keep bit-compat when src_prefix_len=0 (v20k runs must be reproducible).
- Accept externally provided rotary tables: input (qcos,qsin) applied to q,k after
  l2 norm (port apply_rotary_pairs from lact_nvs/lact_ttt_cam.py); hidden
  (hcos,hsin) replacing the carrier-trick grid phases when provided.
- Class kwargs: ttt_input_rope=False, cam_phase_mode: none|plucker,
  ttt_learnable_freqs reused for both sites.
- Attention: implement the split described above (src windows separately; existing
  path for tgt with offsets).

### 4. MODIFY minVid/models/video_latent_flow_matching_ar.py (+ wan wrapper)
- Pair batch path: VAE-encode both videos (src clean, no noise); prepend src
  latents; existing interleave for tgt; pass src_prefix_len down.
- cam_encoder variant (flag use_cam_encoder): per WanAttentionBlock add
  cam_encoder=nn.Linear(12,dim) (default init) + cam_projector=nn.Linear(dim,dim)
  (identity init). Per latent frame t: rel = inv(c2w_src_cv[t]) @ c2w_tgt_cv[t]
  (canonical CV frames), 3x4 flattened; src frames use identity pose. hidden +=
  cam_projector(cam_encoder(rel12)) broadcast over that frame's tokens, before
  self-attn, every block (ReCamMaster recipe).
- Plucker phases: build once per step fp32; order [src 21 frames || tgt interleave
  frame order] matching the token sequence exactly (the interleave duplicates tgt
  windows as noisy+clean: BOTH copies of a frame get the same phases).
- deterministic_noise path must keep working.

### 5. Configs minVid/configs/ar/abl_ccv_{base,pra,both}.yaml
From abl_video_base.yaml. dataset target multicam_pair_dataset.MultiCamPairDataModule,
data_root /tmp/mcv_clips, num_pairs 2000, index_seed 42. lr 2e-5, 20000 steps,
save_every 2000, deterministic noise on, wandb disabled.
- base: use_cam_encoder=true, no rotary flags.
- pra:  ttt_input_rope=true, ttt_hidden_rope=true, cam_phase_mode=plucker,
        use_cam_encoder=false.
- both: all on.

### 6. run_ccv_sanity.sh + report
GPU 3 only. For each variant: 60 steps, batch 1; log to /tmp/ccv_sanity_<v>.log.
Check: (a) losses finite and broadly decreasing; (b) step-1/2 losses of base vs pra
identical (warmup lr~0) but diverged by step ~3-5 (phases active); (c) trainable
param counts: base ~= +71M (30 x (12x1536 + 1536^2 + biases)) vs v20k; pra ~= +~0
(ladder params only); (d) peak memory fits (v20k uses ~80GB at 60,840 tok; this is
93,600 tok; B200 has 192GB).
Also run the dataset __main__ sanity and paste its output in the report.

## Do NOT
- Touch GPUs 0-2 or any running process. Do not launch the 20k runs.
- Modify lact_nvs/ or lact_llm/ (read-only reference).
- Break v20k reproducibility (all new behavior behind flags/defaults).
